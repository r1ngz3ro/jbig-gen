#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

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

uint8_t stream[0x1000];
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
    memcpy(stream + stream_pos, &header_flags, sizeof(header_flags));
    stream_pos += sizeof(header_flags);
    if (k.page_number_known){
        uint32_t number_of_pages = urand();
        printf("number_of_pages = %u\n", number_of_pages);
        memcpy(stream + stream_pos, &number_of_pages, sizeof(number_of_pages));
        stream_pos += sizeof(number_of_pages);
    }
}

void fill_random_pattern(uint8_t *buf, size_t len);

uint8_t *gensegmentheader(size_t *out_len)
{
    size_t header_cap = 0x10;
    size_t header_pos = 0;
    uint8_t *header_buf = malloc(header_cap);
    if (header_buf == NULL) {
        fprintf(stderr, "malloc failed\n");
        return NULL;
    }
    uint32_t segment_number = urand();
    uint8_t segment_flags = urand() & 0xFF;
    uint8_t segment_type = segment_flags & 0x3F;
    uint32_t refs_out = urand();
    uint32_t R = refs_out;
    // FIXME: cap R to keep output sane; real size depends on the spec only, remove this later
    if (R > 10)
        R = 10;

    size_t retrf_size = (R <= 4) ? 1 : 4 + ( (R + 1 + 7 ) / 8);

    // 7.2.7  this special segment type has a very special pading to a very special case 
    uint32_t segment_data_length = (segment_type == SEG_IMMEDIATE_GENERIC)
                                       ? 0xffffffff
                                       : urand();

    printf("segment number = %u\n", segment_number);

    header_buf = realloc(header_buf, header_pos + sizeof(segment_number));
    if (header_buf == NULL) {
        fprintf(stderr, "realloc failed\n");
        return NULL;
    }
    memcpy(header_buf + header_pos, &segment_number, sizeof(segment_number));
    header_pos += sizeof(segment_number);

    header_buf = realloc(header_buf, header_pos + sizeof(segment_flags));
    if (header_buf == NULL) {
        fprintf(stderr, "realloc failed\n");
        return NULL;
    }
    memcpy(header_buf + header_pos, &segment_flags, sizeof(segment_flags));
    header_pos += sizeof(segment_flags);

    header_buf = realloc(header_buf, header_pos + sizeof(segment_data_length));
    if (header_buf == NULL) {
        fprintf(stderr, "realloc failed\n");
        return NULL;
    }
    memcpy(header_buf + header_pos, &segment_data_length, sizeof(segment_data_length));
    header_pos += sizeof(segment_data_length);

    header_buf = realloc(header_buf, header_pos + retrf_size);
    if (header_buf == NULL) {
        fprintf(stderr, "realloc failed\n");
        return NULL;
    }
    if (R <= 4) {
        header_buf[header_pos++] = urand() & 0xFF;
    } else {
        fill_random_pattern(header_buf + header_pos, retrf_size);
        header_pos += retrf_size;
    }

    printf("segment header generated\n");
    if (out_len)
        *out_len = header_pos;
    return header_buf;
}

void fill_random_pattern(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++)
        buf[i] = urand() & 0xFF;
}

uint8_t *gensegmentdata(void)
{
    static uint8_t data_buf[0x40];
    static size_t data_pos = 0;
    printf("segment data generated\n");
    return data_buf;
}

uint8_t *gensegment(void)
{
    uint8_t *header_buf = gensegmentheader(&g_segment_len);
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
    uint8_t *seg = gensegment();
    if (k.colored_region)
        printf("Colored region\n");
    hexdump(stream, stream_pos);
    bindump(stream[0]);
    printf("--\n");
    hexdump(seg, g_segment_len);
    free(seg);
    return 0;
}
