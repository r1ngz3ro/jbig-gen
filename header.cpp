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

std::vector<uint8_t> gensegmentheader(size_t *out_len)
{
    std::vector<uint8_t> header_buf;
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

    append(header_buf, &segment_number, sizeof(segment_number));
    append(header_buf, &segment_flags, sizeof(segment_flags));
    append(header_buf, &segment_data_length, sizeof(segment_data_length));

    if (retrf_size == 1 ) {

	    // get a rand u8
	uint8_t retention_flags= (urand() & 0xFF) ;
	   // get bits 5-7
	uint8_t refs= rentention_flags & 0xE0;
	  // shift right 
	refs = refs >> 5 ;
	
	// we now adjust retention flags to match ref count
	// we cant have the number of flags indicate more refs than the number indicated by the ref count value we bot eariler from retiontion_flags field 
	if (refs < 4) {
		// zero out bit 4
	 retention_flags &= 0xEF ;
	
	}
	if (refs < 3) {
		// zero out bit 4-3
	 retention_flags &= 0xE7;
	
	}
	if (refs < 2) {
		// zero out bit 4-2
	 retention_flags &= 0xE3;
	
	}
	if (refs < 1) {
		// zero out bit 4-1
	 retention_flags &= 0xE1 ;
	}

	// even if the flags match the ref count , the retention bit could still possibly be set to zero  
	// if the bit is set to zero on bit 0 , all other higher fields should logically be null
	// we try to hold this logic true in the following checks
	if (retention_flags & 0x01 == 0 ) {
		// if bit0 is 0 zero out all bits 0- 4 
	 retention_flags &= 0xE0 ;
	}
	if (retention_flags & 0x02 == 0 ) {
		// if bit1 is 0 zero out all bits 1- 4 
	 retention_flags &= 0xE1 ;
	}
	if (retention_flags & 0x04 == 0 ) {
		// if bit2 is 0 zero out all bits 2- 4 
	 retention_flags &= 0xE3 ;
	}
	if (retention_flags & 0x08 == 0 ) {
		// if bit3 is 0 zero out all bits 3- 4 
	 retention_flags &= 0xE7 ;
	}
	if (retention_flags & 0x10 == 0 ) {
		// if bit4 is 0 zero out  bit  4 
	 retention_flags &= 0xEF ;
	}
	

	if (	
        header_buf.push_back(retention_flags);
    } else {

	uint32_t retention_flags= urand()  ;
	// get first 28 bits
	uint32_t refs= rentention_flags & 0x1FFFFFFF;
	// set bits 31-29 to 1
	 retention_flags |= 0xE0000000  ;
// since the number of refs here is much higher i need some complex func to handle the switch case i did in the short 1 byte case 	

	
// once we finally set up retainment
// WE REALLY NEED TO FIGURE OUT WHAT THE HELL RETAINMENT MEANS

/*
        size_t base = header_buf.size();
        header_buf.resize(base + retrf_size);
	
        fill_random_pattern(header_buf.data() + base, retrf_size);
*/
    }

    printf("segment header generated\n");
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
    std::vector<uint8_t> seg = gensegment();
    if (k.colored_region)
        printf("Colored region\n");
    hexdump(stream.data(), stream_pos);
    bindump(stream[0]);
    printf("--\n");
    hexdump(seg.data(), seg.size());
    return 0;
}
