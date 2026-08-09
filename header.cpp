#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// xorshift32 PRNG. Seeded once from /dev/urandom and never returns to the
// kernel, unlike the previous per-call urandom read which dominated
// generation cost (~5.7s/1k samples).
static uint32_t prng_state;

static uint32_t xorshift32(void)
{
    uint32_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng_state = x;
    return x;
}

// Seeds the PRNG from /dev/urandom; exits on failure.
void urand_init(void)
{
    FILE *f = fopen("/dev/urandom", "r");
    if (f == NULL) {
        perror("/dev/urandom");
        exit(1);
    }
    uint32_t seed;
    if (fread(&seed, sizeof(seed), 1, f) != 1) {
        perror("fread /dev/urandom");
        exit(1);
    }
    fclose(f);
    if (seed == 0)
        seed = 0x6D2B79F5u; // xorshift state must be nonzero
    prng_state = seed;
}

uint32_t urand(void)
{
    return xorshift32();
}

static void append(std::vector<uint8_t> &v, const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++)
        v.push_back(b[i]);
}

// JBIG2 is big-endian: every multi-byte field is most-significant byte first.
static void put_be32(std::vector<uint8_t> &v, uint32_t x)
{
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}

static void put_be16(std::vector<uint8_t> &v, uint16_t x)
{
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}

typedef enum {
    ORG_RANDOM_ACCESS,
    ORG_SEQUENTIAL,
    // not a real JBIG2 org type; embedded org has no JBIG2 file header (D.4)
    ORG_EMBEDDED
} Organization;

typedef struct {
    bool page_number_known;
    bool use_12_AT;
    bool colored_region;
} Knubs;

typedef enum {
    SEG_SYMBOL_DICTIONARY = 0,
    SEG_INTERMEDIATE_TEXT = 4,
    SEG_IMMEDIATE_TEXT = 6,
    SEG_IMMEDIATE_LOSSLESS_TEXT = 7,
    SEG_PATTERN_DICTIONARY = 16,
    SEG_INTERMEDIATE_HALFTONE = 20,
    SEG_IMMEDIATE_HALFTONE = 22,
    SEG_IMMEDIATE_LOSSLESS_HALFTONE = 23,
    SEG_INTERMEDIATE_GENERIC = 36,
    SEG_IMMEDIATE_GENERIC = 38,
    SEG_IMMEDIATE_LOSSLESS_GENERIC = 39,
    SEG_INTERMEDIATE_GENERIC_REFINEMENT = 40,
    SEG_IMMEDIATE_GENERIC_REFINEMENT = 42,
    SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT = 43,
    SEG_PAGE_INFORMATION = 48,
    SEG_END_OF_PAGE = 49,
    SEG_END_OF_STRIPE = 50,
    SEG_END_OF_FILE = 51,
    SEG_PROFILES = 52,
    SEG_TABLES = 53,
    SEG_COLOUR_PALETTE = 54,
    SEG_EXTENSION = 62
} SegmentType;

Knubs knubs(void)
{
    Knubs k;
    k.page_number_known = urand() % 2;
    k.use_12_AT = urand() % 2;
    k.colored_region = urand() % 2;
    return k;
}

std::vector<uint8_t> stream;

size_t g_segment_len = 0;

// Two pointer streams filled from gensegment()'s output: one for the
// header parts, one for the data parts, in generation order.
// Note: order of creation is preserved, but Annex D.1/D.2 require segments
// in increasing segment-number order, whereas gensegmentheader() picks the
// segment number at random. TODO: assign increasing segment numbers.
std::vector<std::vector<uint8_t> *> header_streams;
std::vector<std::vector<uint8_t> *> data_streams;

Organization choose_organisation(void)
{
    return (Organization)(urand() % 3);
}

void genheader(Organization org, Knubs k)
{
    if (org == ORG_EMBEDDED) {
        printf("embedded organisation has no file header (D.4)\n");
        return;
    }

    // D.4.2: file header flags.
    uint8_t header_flags = (org == ORG_SEQUENTIAL) ? 0x01 : 0x00;
    if (!k.page_number_known)
        header_flags |= 0x02;
    if (k.use_12_AT)
        header_flags |= 0x04;
    if (k.colored_region)
        header_flags |= 0x08;

    printf("header_flags = 0x%02X (%u)\n", header_flags, header_flags);
    stream.push_back(header_flags);

    // D.4.3: number of pages; omitted when the count is unknown.
    if (k.page_number_known) {
        uint32_t number_of_pages = 1 + (urand() % 8);
        printf("number_of_pages = %u\n", number_of_pages);
        put_be32(stream, number_of_pages);
    }
}

void fill_random_pattern(uint8_t *buf, size_t len);
void hexdump(const uint8_t *buf, size_t len);

// Writes the retain-bit field. R = number of referred-to segments.
// Meaningful bits: bit 0 = retain-this-segment, bits 1..R = retain bits
// for referred-to segments 1..R, in that order. Bits beyond R must be 0.
void write_retention_bits(std::vector<uint8_t> &out, uint32_t R)
{
    size_t nbits  = R + 1;
    size_t nbytes = (nbits + 7) / 8;
    std::vector<uint8_t> bits(nbytes, 0);   // zero-init handles the "must be 0" part for free
    for (size_t i = 0; i < nbits; i++) {
        if (urand() & 1)
            bits[i / 8] |= (1u << (i % 8));
    }
    append(out, bits.data(), bits.size());
}

// Produces a segment header. Field order (Figure 27 / 7.2.2-7.2.7):
//   segment number, header flags, referred-to count + retention flags,
//   referred-to segment numbers, page association, segment data length.
std::vector<uint8_t> gensegmentheader(uint8_t segment_type, uint32_t data_len, size_t *out_len)
{
    std::vector<uint8_t> hdr;

    uint32_t segment_number = urand();
    uint32_t R = urand() % 26;              // 0..25 referred-to segments
    bool big_page = urand() & 1;            // page association field size

    // 7.2.3: segment header flags.
    uint8_t flags = segment_type & 0x3F;    // bits 0-5: segment type
    if (big_page)
        flags |= 0x40;                      // bit 6: 4-byte page association
    if (urand() & 1)
        flags |= 0x80;                      // bit 7: deferred non-retain

    // 7.2.7: immediate generic region may carry an unknown data length.
    uint32_t seg_len = (segment_type == SEG_IMMEDIATE_GENERIC)
                           ? 0xFFFFFFFFu
                           : data_len;

    printf("segment number = %u, type = %u, R = %u\n", segment_number, segment_type, R);

    put_be32(hdr, segment_number);
    hdr.push_back(flags);

    // 7.2.4: referred-to segment count and retention flags.
    // short form: one byte, top 3 bits = R, bits 0-4 = retain bits.
    std::vector<uint8_t> retain;
    write_retention_bits(retain, R);
    if (R <= 4) {
        hdr.push_back((uint8_t)((R << 5) | (retain[0] & 0x1F)));
    } else {
        // long form: bits 29-31 = 7, low 29 bits = R, then ceil((R+1)/8) bytes.
        put_be32(hdr, 0xE0000000u | R);
        append(hdr, retain.data(), retain.size());
    }

    // 7.2.5: referred-to segment numbers. A segment may only refer to
    // lower-numbered segments; its field size depends on its own number.
    for (uint32_t i = 0; i < R; i++) {
        uint32_t ref = segment_number ? (urand() % segment_number) : 0;
        if (segment_number <= 256)
            hdr.push_back((uint8_t)ref);
        else if (segment_number <= 65536)
            put_be16(hdr, (uint16_t)ref);
        else
            put_be32(hdr, ref);
    }

    // 7.2.6: page association. 0 = not associated with any page; 1 is the first page.
    if (urand() & 1) {
        if (big_page)
            put_be32(hdr, 0);
        else
            hdr.push_back(0);
    } else {
        uint32_t page = 1 + (urand() & (big_page ? 0x0FFFFFFFu : 0xFFu));
        if (big_page)
            put_be32(hdr, page);
        else
            hdr.push_back((uint8_t)page);
    }

    put_be32(hdr, seg_len);

    printf("segment header generated (%zu bytes)\n", hdr.size());
    hexdump(hdr.data(), hdr.size());
    if (out_len)
        *out_len = hdr.size();
    return hdr;
}

void fill_random_pattern(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++)
        buf[i] = urand() & 0xFF;
}

// A segment-data handler builds the data part for one segment type.
// Table is indexed by SegmentType; entries that are nullptr are
// "unimplemented" and gensegmentdata() falls back to random bytes.
typedef std::vector<uint8_t> (*SegHandler)(void);

std::vector<uint8_t> gen_segment_page_info(void);
std::vector<uint8_t> gen_segment_extension(void);
std::vector<uint8_t> gen_segment_pattern_dict(void);
std::vector<uint8_t> gen_segment_region_info(void);

static const size_t SEG_HANDLER_COUNT = 63;
static SegHandler seg_handlers[SEG_HANDLER_COUNT] = { nullptr };

void init_seg_handlers(void)
{
    seg_handlers[SEG_PAGE_INFORMATION] = gen_segment_page_info;
    seg_handlers[SEG_EXTENSION] = gen_segment_extension;
    seg_handlers[SEG_PATTERN_DICTIONARY] = gen_segment_pattern_dict;

    seg_handlers[SEG_INTERMEDIATE_TEXT] = gen_segment_region_info;
    seg_handlers[SEG_IMMEDIATE_TEXT] = gen_segment_region_info;
    seg_handlers[SEG_IMMEDIATE_LOSSLESS_TEXT] = gen_segment_region_info;
    seg_handlers[SEG_INTERMEDIATE_HALFTONE] = gen_segment_region_info;
    seg_handlers[SEG_IMMEDIATE_HALFTONE] = gen_segment_region_info;
    seg_handlers[SEG_IMMEDIATE_LOSSLESS_HALFTONE] = gen_segment_region_info;
    seg_handlers[SEG_INTERMEDIATE_GENERIC] = gen_segment_region_info;
    seg_handlers[SEG_IMMEDIATE_GENERIC] = gen_segment_region_info;
    seg_handlers[SEG_IMMEDIATE_LOSSLESS_GENERIC] = gen_segment_region_info;
    seg_handlers[SEG_INTERMEDIATE_GENERIC_REFINEMENT] = gen_segment_region_info;
    seg_handlers[SEG_IMMEDIATE_GENERIC_REFINEMENT] = gen_segment_region_info;
    seg_handlers[SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT] = gen_segment_region_info;
}

// 7.4.1: region segment information field (Figure 30). This common
// 17-byte prefix opens the data part of every region segment (text,
// halftone, generic, generic refinement). Field order:
//   width, height, X location, Y location, flags.
std::vector<uint8_t> gen_segment_region_info(void)
{
    std::vector<uint8_t> d;
    // Keep the size/location fields within 16 bits so regions stay
    // plausible next to the page size reported by gen_segment_page_info()
    // (itself capped at 1 + urand()%0x10000).
    put_be32(d, 1 + (urand() % 0x10000));     // 7.4.1.1: bitmap width
    put_be32(d, 1 + (urand() % 0x10000));     // 7.4.1.2: bitmap height
    put_be32(d, urand() & 0xFFFF);            // 7.4.1.3: X location
    put_be32(d, urand() & 0xFFFF);            // 7.4.1.4: Y location
    // 7.4.1.5 flags: bits 0-2 external combination operator (0 OR,
    // 1 AND, 2 XOR, 3 XNOR, 4 REPLACE); bit 3 COLEXTFLAG; bits 4-7
    // reserved, must be 0.
    bool color = (urand() & 1) != 0;
    uint8_t flags = color ? 4 : (uint8_t)(urand() % 5); // COLEXTFLAG -> REPLACE (Note 3)
    if (color)
        flags |= 0x08;
    d.push_back(flags);
    printf("region-info handler (%zu bytes)\n", d.size());
    return d;
}

// 4.4.1: page information data part.
std::vector<uint8_t> gen_segment_page_info(void)
{
    std::vector<uint8_t> d;
    put_be32(d, 1 + (urand() % 0x10000));     // page width
    put_be32(d, 1 + (urand() % 0x10000));     // page height
    put_be32(d, 1 + (urand() % 300));         // x resolution
    put_be32(d, 1 + (urand() % 300));         // y resolution
    d.push_back((urand() & 1) ? 0x80 : 0x00); // page flags: default pixel value
    d.push_back((uint8_t)(urand() % 0xFF));   // stripe size
    printf("page-info handler (%zu bytes)\n", d.size());
    return d;
}

// 4.4.8: extension segment data part (2-byte extension type).
std::vector<uint8_t> gen_segment_extension(void)
{
    std::vector<uint8_t> d;
    put_be16(d, (uint16_t)(urand() & 0xFFFE));
    printf("extension handler (%zu bytes)\n", d.size());
    return d;
}

// 4.4.2: pattern dictionary segment data header (Figure 41).
std::vector<uint8_t> gen_segment_pattern_dict(void)
{
    std::vector<uint8_t> d;
    bool mmr = (urand() & 1) != 0;
    // 4.4.2.1.1 flag: bit 0 HDMMR, bits 1-2 HDTEMPLATE; HDTEMPLATE must be
    // 0 when HDMMR is set. Bits 3-7 reserved, always 0.
    uint8_t flags = mmr ? 0x01 : 0x00;
    if (!mmr)
        flags |= (uint8_t)((urand() % 4) << 1);
    d.push_back(flags);
    d.push_back((uint8_t)(1 + (urand() % 0xFF)));   // HDPW, must be > 0
    d.push_back((uint8_t)(1 + (urand() % 0xFF)));   // HDPH, must be > 0
    put_be32(d, 1 + (urand() % 0x10000));           // GRAYMAX = npatterns - 1
    printf("pattern-dictionary handler (%zu bytes)\n", d.size());
    return d;
}

std::vector<uint8_t> gensegmentdata(uint8_t segment_type, uint32_t max_len)
{
    SegHandler handler = nullptr;
    switch (segment_type) {
    case SEG_SYMBOL_DICTIONARY:
        handler = seg_handlers[SEG_SYMBOL_DICTIONARY];
        break;
    case SEG_INTERMEDIATE_TEXT:
        handler = seg_handlers[SEG_INTERMEDIATE_TEXT];
        break;
    case SEG_IMMEDIATE_TEXT:
        handler = seg_handlers[SEG_IMMEDIATE_TEXT];
        break;
    case SEG_IMMEDIATE_LOSSLESS_TEXT:
        handler = seg_handlers[SEG_IMMEDIATE_LOSSLESS_TEXT];
        break;
    case SEG_PATTERN_DICTIONARY:
        handler = seg_handlers[SEG_PATTERN_DICTIONARY];
        break;
    case SEG_INTERMEDIATE_HALFTONE:
        handler = seg_handlers[SEG_INTERMEDIATE_HALFTONE];
        break;
    case SEG_IMMEDIATE_HALFTONE:
        handler = seg_handlers[SEG_IMMEDIATE_HALFTONE];
        break;
    case SEG_IMMEDIATE_LOSSLESS_HALFTONE:
        handler = seg_handlers[SEG_IMMEDIATE_LOSSLESS_HALFTONE];
        break;
    case SEG_INTERMEDIATE_GENERIC:
        handler = seg_handlers[SEG_INTERMEDIATE_GENERIC];
        break;
    case SEG_IMMEDIATE_GENERIC:
        handler = seg_handlers[SEG_IMMEDIATE_GENERIC];
        break;
    case SEG_IMMEDIATE_LOSSLESS_GENERIC:
        handler = seg_handlers[SEG_IMMEDIATE_LOSSLESS_GENERIC];
        break;
    case SEG_INTERMEDIATE_GENERIC_REFINEMENT:
        handler = seg_handlers[SEG_INTERMEDIATE_GENERIC_REFINEMENT];
        break;
    case SEG_IMMEDIATE_GENERIC_REFINEMENT:
        handler = seg_handlers[SEG_IMMEDIATE_GENERIC_REFINEMENT];
        break;
    case SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT:
        handler = seg_handlers[SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT];
        break;
    case SEG_PAGE_INFORMATION:
        handler = seg_handlers[SEG_PAGE_INFORMATION];
        break;
    case SEG_END_OF_PAGE:
        handler = seg_handlers[SEG_END_OF_PAGE];
        break;
    case SEG_END_OF_STRIPE:
        handler = seg_handlers[SEG_END_OF_STRIPE];
        break;
    case SEG_END_OF_FILE:
        handler = seg_handlers[SEG_END_OF_FILE];
        break;
    case SEG_PROFILES:
        handler = seg_handlers[SEG_PROFILES];
        break;
    case SEG_TABLES:
        handler = seg_handlers[SEG_TABLES];
        break;
    case SEG_COLOUR_PALETTE:
        handler = seg_handlers[SEG_COLOUR_PALETTE];
        break;
    case SEG_EXTENSION:
        handler = seg_handlers[SEG_EXTENSION];
        break;
    default:
        break;
    }

    // Unimplemented (null) handlers are skipped: fall back to random data.
    if (handler != nullptr) {
        std::vector<uint8_t> data = handler();
        return data;
    }

    uint32_t len = urand() % (max_len + 1);
    std::vector<uint8_t> data(len);
    fill_random_pattern(data.data(), len);
    printf("segment data generated (%u bytes)\n", len);
    return data;
}

std::vector<std::vector<uint8_t> *> gensegment(void)
{
    static const uint8_t types[] = {
        SEG_SYMBOL_DICTIONARY, SEG_INTERMEDIATE_TEXT, SEG_IMMEDIATE_TEXT,
        SEG_IMMEDIATE_LOSSLESS_TEXT, SEG_PATTERN_DICTIONARY,
        SEG_INTERMEDIATE_HALFTONE, SEG_IMMEDIATE_HALFTONE,
        SEG_IMMEDIATE_LOSSLESS_HALFTONE, SEG_INTERMEDIATE_GENERIC,
        SEG_IMMEDIATE_GENERIC, SEG_IMMEDIATE_LOSSLESS_GENERIC,
        SEG_INTERMEDIATE_GENERIC_REFINEMENT, SEG_IMMEDIATE_GENERIC_REFINEMENT,
        SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT, SEG_PAGE_INFORMATION,
        SEG_END_OF_PAGE, SEG_END_OF_STRIPE, SEG_END_OF_FILE,
        SEG_PROFILES, SEG_TABLES, SEG_COLOUR_PALETTE, SEG_EXTENSION
    };
    uint8_t type = types[urand() % (sizeof(types) / sizeof(types[0]))];

    // EOF / end-of-page / end-of-stripe segments carry no data.
    bool has_data = type != SEG_END_OF_PAGE && type != SEG_END_OF_STRIPE
                    && type != SEG_END_OF_FILE;

    std::vector<uint8_t> data;
    if (has_data)
        data = gensegmentdata(type, 256);
    uint32_t data_len = (uint32_t)data.size();

    if (type == SEG_IMMEDIATE_GENERIC) {
        // 7.2.7: with an unknown data length the data part must end with the
        // template-coding terminator 0xFF 0xAC plus a 4-byte row count.
        static const uint8_t term[2] = { 0xFF, 0xAC };
        append(data, term, sizeof(term));
        put_be32(data, urand());
        data_len = (uint32_t)data.size();
    }

    std::vector<uint8_t> *hdr = new std::vector<uint8_t>(
        gensegmentheader(type, data_len, &g_segment_len));

    std::vector<std::vector<uint8_t> *> parts;
    parts.push_back(hdr);
    parts.push_back(has_data ? new std::vector<uint8_t>(data) : nullptr);
    return parts;
}

void hexdump(const uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        printf("%02X ", buf[i]);
        if ((i + 1) % 16 == 0)
            printf("\n");
    }
    if (len % 16 != 0)
        printf("\n");
}

void bindump(uint8_t byte)
{
    for (int i = 7; i >= 0; i--)
        printf("%d", (byte >> i) & 1);
    printf("\n");
}

void serialize_out(const uint8_t *buf, size_t len, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        perror("fopen");
        return;
    }
    if (fwrite(buf, 1, len, f) != len)
        perror("fwrite");
    fclose(f);
}

// Appends the two pointer streams into `stream` in the order dictated by
// the file organization (Annex D.1 / D.2 / D.3).
void assemble_org_order(Organization org)
{
    for (size_t i = 0; i < header_streams.size(); i++) {
        if (header_streams[i])
            append(stream, header_streams[i]->data(), header_streams[i]->size());
        if (org == ORG_SEQUENTIAL || org == ORG_EMBEDDED)
            if (data_streams[i])
                append(stream, data_streams[i]->data(), data_streams[i]->size());
    }
    // Random access: every segment header first, then all segment data.
    if (org == ORG_RANDOM_ACCESS)
        for (size_t i = 0; i < data_streams.size(); i++)
            if (data_streams[i])
                append(stream, data_streams[i]->data(), data_streams[i]->size());
}

void free_streams(void)
{
    for (size_t i = 0; i < header_streams.size(); i++)
        delete header_streams[i];
    for (size_t i = 0; i < data_streams.size(); i++)
        delete data_streams[i];
    header_streams.clear();
    data_streams.clear();
}

// Master init: everything that must happen before generation starts.
void init_all(void)
{
    urand_init();
    init_seg_handlers();
}

int main(int argc, char **argv)
{
    const char *out_path = (argc > 1) ? argv[1] : "out.jb2";

    init_all();
    Organization org = choose_organisation();
    switch (org) {
    case ORG_RANDOM_ACCESS:
        printf("Random-access\n");
        break;
    case ORG_SEQUENTIAL:
        printf("Sequential\n");
        break;
    case ORG_EMBEDDED:
        printf("Embedded\n");
        break;
    }
    Knubs k = knubs();
    genheader(org, k);

    size_t nseg = 1 + (urand() % 4);
    for (size_t i = 0; i < nseg; i++) {
        std::vector<std::vector<uint8_t> *> seg = gensegment();
        header_streams.push_back(seg[0]);
        data_streams.push_back(seg[1]);
    }

    assemble_org_order(org);

    serialize_out(stream.data(), stream.size(), out_path);

    if (k.colored_region)
        printf("Colored region\n");
    hexdump(stream.data(), stream.size());
    if (!stream.empty())
        bindump(stream[0]);
    for (size_t i = 0; i < header_streams.size(); i++) {
        printf("--\n");
        hexdump(header_streams[i]->data(), header_streams[i]->size());
        if (data_streams[i]) {
            printf("--\n");
            hexdump(data_streams[i]->data(), data_streams[i]->size());
        }
    }

    free_streams();
    return 0;
}
