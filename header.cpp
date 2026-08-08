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

// FIXME endianness: append() copies host-order memory as-is (little-endian on x86),
// but the JBIG2 spec uses big-endian fields. Wrap appends in explicit BE writers later.

typedef enum {
    ORG_RANDOM_ACCESS,
    ORG_SEQUENTIAL,
    // not a real JBIG2 org type; reserved for future use
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
size_t stream_pos = 0;
size_t g_segment_len = 0;

Organization choose_organisation(void)
{
    return (Organization)(urand() % 3);
}

void genheader(Organization org, Knubs k)
{
    uint8_t header_flags = 0;

    if (org == ORG_SEQUENTIAL)
        header_flags |= 1;
    if (org == ORG_RANDOM_ACCESS)
        header_flags = 0;

    if (!k.page_number_known) {
        header_flags |= 2;
    } 

    if (k.use_12_AT)
        header_flags |= 4;

    if (k.colored_region)
        header_flags |= 8;

    printf("header_flags = 0x%02X (%u)\n", header_flags, header_flags);
    append(stream, &header_flags, sizeof(header_flags));
    stream_pos += sizeof(header_flags);
    if (k.page_number_known){
        uint32_t number_of_pages = urand();
        printf("number_of_pages = %u\n", number_of_pages);
        append(stream, &number_of_pages, sizeof(number_of_pages));
        stream_pos += sizeof(number_of_pages);
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

std::vector<uint8_t> gensegmentheader(size_t *out_len)
{
    std::vector<uint8_t> header_buf;
    uint32_t segment_number = urand();
    uint8_t segment_flags = urand() & 0xFF;
    uint8_t segment_type = segment_flags & 0x3F;
    uint32_t refs_out = urand();
    uint32_t R = refs_out;
    // FIXME: cap R to keep output sane; real size depends on the spec only, remove this later
    if (R > 25)
        R = 25;

    // 7.2.7  this special segment type has a very special pading to a very special case 
    uint32_t segment_data_length = (segment_type == SEG_IMMEDIATE_GENERIC)
                                       ? 0xffffffff
                                       : urand();

    printf("segment number = %u\n", segment_number);

    append(header_buf, &segment_number, sizeof(segment_number));
    append(header_buf, &segment_flags, sizeof(segment_flags));
    append(header_buf, &segment_data_length, sizeof(segment_data_length));

    
    if (R <= 4) {
    // short form: 1 byte total. Top 3 bits = R, bottom bits = retain field.
    std::vector<uint8_t> retain;
    write_retention_bits(retain, R);      // produces exactly 1 byte for R<=4
    uint8_t b = (uint8_t)((R << 5) | (retain[0] & 0x1F));
    header_buf.push_back(b);
} else {
    // long form: 4-byte count word (top 3 bits = 0b111, low 29 bits = R),
    // followed by ceil((R+1)/8) bytes of retain bits.
    uint32_t count_word = 0xE0000000u | (R & 0x1FFFFFFFu);
    append(header_buf, &count_word, sizeof(count_word));   // still needs BE swap eventually
    write_retention_bits(header_buf, R);
}

    printf("segment header generated\n");
    hexdump(header_buf.data(), header_buf.size());
    if (out_len)
        *out_len = header_buf.size();
    return header_buf;
}

void fill_random_pattern(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++)
        buf[i] = urand() & 0xFF;
}

std::vector<uint8_t> gensegmentdata(void)
{
    static std::vector<uint8_t> data_buf;
    printf("segment data generated\n");
    return data_buf;
}

std::vector<uint8_t> gensegment(void)
{
    std::vector<uint8_t> header_buf = gensegmentheader(&g_segment_len);
    gensegmentdata();
    return header_buf;
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

void serialize_out(const uint8_t *buf, size_t len)
{
    FILE *f = fopen("out.jb2", "wb");
    if (f == NULL) {
        perror("fopen out.jb2");
        return;
    }
    if (fwrite(buf, 1, len, f) != len)
        perror("fwrite out.jb2");
    fclose(f);
}

int main(void)
{
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
    serialize_out(stream.data(), stream_pos);
    std::vector<uint8_t> seg = gensegment();
    append(stream, seg.data(), seg.size());
    stream_pos += seg.size();
    if (k.colored_region)
        printf("Colored region\n");
    hexdump(stream.data(), stream_pos);
    bindump(stream[0]);
    printf("--\n");
    hexdump(seg.data(), seg.size());
    return 0;
}
