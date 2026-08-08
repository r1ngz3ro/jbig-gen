#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

uint32_t urand(void)
{
    uint32_t v;
    FILE *f = fopen("/dev/urandom", "r");
    if (f == NULL) {
        perror("/dev/urandom");
        exit(1);
    }
    if (fread(&v, sizeof(v), 1, f) != 1) {
        perror("fread /dev/urandom");
        exit(1);
    }
    fclose(f);
    return v;
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

std::vector<uint8_t> gensegmentdata(uint32_t max_len)
{
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
        data = gensegmentdata(256);
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

int main(int argc, char **argv)
{
    const char *out_path = (argc > 1) ? argv[1] : "out.jb2";

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