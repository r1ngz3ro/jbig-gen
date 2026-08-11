#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
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

// Packs bits MSB-first into bytes (5.4.1), for bit-level fields such as
// Huffman code table lines that aren't byte-aligned.
struct BitWriter {
    std::vector<uint8_t> bytes;
    uint8_t cur = 0;
    int nbits = 0;
};

static void bw_put_bits(BitWriter &bw, uint32_t value, int width)
{
    for (int i = width - 1; i >= 0; i--) {
        bw.cur = (uint8_t)((bw.cur << 1) | ((value >> i) & 1));
        if (++bw.nbits == 8) {
            bw.bytes.push_back(bw.cur);
            bw.cur = 0;
            bw.nbits = 0;
        }
    }
}

// Flushes any partial byte, padding the low-order bits with 0.
static void bw_finish(BitWriter &bw)
{
    if (bw.nbits > 0) {
        bw.cur = (uint8_t)(bw.cur << (8 - bw.nbits));
        bw.bytes.push_back(bw.cur);
        bw.cur = 0;
        bw.nbits = 0;
    }
}

// Adaptive template pixel pairs aren't legal at every signed-byte value:
// decoders (and 6.3.5.3/Figure 7) restrict them to a "field" relative to
// the current pixel. Two roles exist:
//   "primary" AT pixels (SDATn, GBATn, and the first of each refinement
//   pair) drive a causal template, so must lie at or before the current
//   pixel in raster order: Y in [-128,0], X in [-128,127], and X < 0 when
//   Y == 0 (Y == 0, X >= 0 would be the current or a future pixel).
//   "reference" AT pixels (the second of each refinement pair, indexing
//   into the separate reference bitmap) have no causality constraint:
//   X and Y are each freely in [-128,127].
struct AtPixel {
    int x, y;
};

static AtPixel write_primary_at_pixel(std::vector<uint8_t> &d)
{
    int y = -(int)(urand() % 129);                          // -128..0
    int x = (y == 0) ? -(int)(1 + urand() % 128)             // -128..-1
                      : -128 + (int)(urand() % 256);         // -128..127
    d.push_back((uint8_t)(int8_t)x);
    d.push_back((uint8_t)(int8_t)y);
    return { x, y };
}

static AtPixel write_reference_at_pixel(std::vector<uint8_t> &d)
{
    int x = -128 + (int)(urand() % 256);
    int y = -128 + (int)(urand() % 256);
    d.push_back((uint8_t)(int8_t)x);
    d.push_back((uint8_t)(int8_t)y);
    return { x, y };
}

// Writes a caller-supplied (not drawn) AT pixel pair. Real decoders carry a
// second, optimized decode routine for generic/refinement regions that only
// triggers when every AT pixel exactly equals the spec's nominal default
// (6.2.5.3 Table 6 for GBTEMPLATE, 6.3.5.3 for GRTEMPLATE 0) -- since
// write_primary_at_pixel()/write_reference_at_pixel() draw uniformly at
// random, that routine is astronomically unlikely to ever run. Callers that
// want the nominal value some fraction of the time use this instead.
static AtPixel write_nominal_at_pixel(std::vector<uint8_t> &d, int x, int y)
{
    d.push_back((uint8_t)(int8_t)x);
    d.push_back((uint8_t)(int8_t)y);
    return { x, y };
}

// 6.2.5.3 Table 6 nominal AT pixel positions, indexed [GBTEMPLATE][pair];
// only pair 0 is meaningful for templates 1-3 (single AT pixel each).
static const AtPixel GB_NOMINAL_AT[4][4] = {
    { { 3, -1 }, { -3, -1 }, { 2, -2 }, { -2, -2 } },
    { { 3, -1 }, {}, {}, {} },
    { { 2, -1 }, {}, {}, {} },
    { { 2, -1 }, {}, {}, {} },
};

// Huffman table selector pickers, for the structural (non-real-content)
// symbol dictionary / text region paths. A selector value of 3 means "use
// a user-supplied table," which 7.4.2.1.6/7.4.3.1.6 require to be backed
// by a distinct referred-to tables segment, consumed from `pool` in order
// and recorded into `refs` so the caller's header stays consistent with
// its own flags. `pool` holds candidate tables-segment numbers not yet
// claimed by an earlier selector in the same segment.

// For the family where legal values are 0, 1, or 3 (2 is reserved).
static uint8_t pick_sel_013(std::vector<uint32_t> &pool, std::vector<uint32_t> &refs)
{
    if (!pool.empty() && (urand() & 1)) {
        refs.push_back(pool.front());
        pool.erase(pool.begin());
        return 3;
    }
    return (uint8_t)(urand() & 1);
}

// For the family where all of 0, 1, 2 are legal standard-table values.
static uint8_t pick_sel_0123(std::vector<uint32_t> &pool, std::vector<uint32_t> &refs)
{
    if (!pool.empty() && (urand() & 1)) {
        refs.push_back(pool.front());
        pool.erase(pool.begin());
        return 3;
    }
    return (uint8_t)(urand() % 3);
}

// For single-bit "use a user-supplied table" flags (e.g. SDHUFFBMSIZE).
static bool pick_sel_bit(std::vector<uint32_t> &pool, std::vector<uint32_t> &refs)
{
    if (!pool.empty() && (urand() & 1)) {
        refs.push_back(pool.front());
        pool.erase(pool.begin());
        return true;
    }
    return false;
}

// Standard Huffman tables B.1-B.15 (Annex B.5): pre-assigned prefix codes
// (already run through the B.3 canonical-code algorithm), one row per
// table line. val_func 2 marks the "lower range" line, where the decoded
// value is RANGELOW - offset instead of RANGELOW + offset (B.4 step 4);
// OOB_VAL marks the out-of-band line, where val/val_func/range_bits are
// unused. Row data transcribed from the ITU-T T.88 reference encoder
// (JBIG2 Sample Software, Jb2_T4T6Lapper.h, Tables A-O => B.1-B.15),
// reproduced here under its BSD-style ITU/ISO reference-software license.
struct StdHuffLine {
    int32_t code;
    int32_t code_len;
    int32_t range_bits;
    int32_t val;
    int32_t val_func;
};
static const int32_t OOB_VAL = INT32_MIN;

// Pointer+count wrapper so a ternary picking between two differently-sized
// standard tables (e.g. `cond ? STD_TABLE(HUFF_B5) : STD_TABLE(HUFF_B4)`)
// is a single well-typed value instead of two arrays decaying to
// incompatible pointer types.
struct StdHuffTable {
    const StdHuffLine *rows;
    size_t n;
};
#define STD_TABLE(x) StdHuffTable{ (x), sizeof(x) / sizeof((x)[0]) }

static const StdHuffLine HUFF_B1[] = {
    { 0x00, 1, 4, 0, 0 }, { 0x02, 2, 8, 16, 0 }, { 0x06, 3, 16, 272, 0 },
    { 0x07, 3, 32, 65808, 0 },
};
static const StdHuffLine HUFF_B2[] = {
    { 0x00, 1, 0, 0, 0 }, { 0x02, 2, 0, 1, 0 }, { 0x06, 3, 0, 2, 0 },
    { 0x0e, 4, 3, 3, 0 }, { 0x1e, 5, 6, 11, 0 }, { 0x3e, 6, 32, 75, 0 },
    { 0x3f, 6, 0, OOB_VAL, 0 },
};
static const StdHuffLine HUFF_B3[] = {
    { 0xff, 8, 32, -257, 2 }, { 0xfe, 8, 8, -256, 0 }, { 0x00, 1, 0, 0, 0 },
    { 0x02, 2, 0, 1, 0 }, { 0x06, 3, 0, 2, 0 }, { 0x0e, 4, 3, 3, 0 },
    { 0x1e, 5, 6, 11, 0 }, { 0x7e, 7, 32, 75, 0 }, { 0x3e, 6, 0, OOB_VAL, 0 },
};
static const StdHuffLine HUFF_B4[] = {
    { 0x00, 1, 0, 1, 0 }, { 0x02, 2, 0, 2, 0 }, { 0x06, 3, 0, 3, 0 },
    { 0x0e, 4, 3, 4, 0 }, { 0x1e, 5, 6, 12, 0 }, { 0x1f, 5, 32, 76, 0 },
};
static const StdHuffLine HUFF_B5[] = {
    { 0x7f, 7, 32, -256, 2 }, { 0x7e, 7, 8, -255, 0 }, { 0x00, 1, 0, 1, 0 },
    { 0x02, 2, 0, 2, 0 }, { 0x06, 3, 0, 3, 0 }, { 0x0e, 4, 3, 4, 0 },
    { 0x1e, 5, 6, 12, 0 }, { 0x3e, 6, 32, 76, 0 },
};
static const StdHuffLine HUFF_B6[] = {
    { 0x3e, 6, 32, -2049, 2 }, { 0x1c, 5, 10, -2048, 0 }, { 0x08, 4, 9, -1024, 0 },
    { 0x09, 4, 8, -512, 0 }, { 0x0a, 4, 7, -256, 0 }, { 0x1d, 5, 6, -128, 0 },
    { 0x1e, 5, 5, -64, 0 }, { 0x0b, 4, 5, -32, 0 }, { 0x00, 2, 7, 0, 0 },
    { 0x02, 3, 7, 128, 0 }, { 0x03, 3, 8, 256, 0 }, { 0x0c, 4, 9, 512, 0 },
    { 0x0d, 4, 10, 1024, 0 }, { 0x3f, 6, 32, 2048, 0 },
};
static const StdHuffLine HUFF_B7[] = {
    { 0x1e, 5, 32, -1025, 2 }, { 0x08, 4, 9, -1024, 0 }, { 0x00, 3, 8, -512, 0 },
    { 0x09, 4, 7, -256, 0 }, { 0x1a, 5, 6, -128, 0 }, { 0x1b, 5, 5, -64, 0 },
    { 0x0a, 4, 5, -32, 0 }, { 0x0b, 4, 5, 0, 0 }, { 0x1c, 5, 5, 32, 0 },
    { 0x1d, 5, 6, 64, 0 }, { 0x0c, 4, 7, 128, 0 }, { 0x01, 3, 8, 256, 0 },
    { 0x02, 3, 9, 512, 0 }, { 0x03, 3, 10, 1024, 0 }, { 0x1f, 5, 32, 2048, 0 },
};
static const StdHuffLine HUFF_B8[] = {
    { 0x1fe, 9, 32, -16, 2 }, { 0x0fc, 8, 3, -15, 0 }, { 0x1fc, 9, 1, -7, 0 },
    { 0x0fd, 8, 1, -5, 0 }, { 0x1fd, 9, 0, -3, 0 }, { 0x07c, 7, 0, -2, 0 },
    { 0x00a, 4, 0, -1, 0 }, { 0x000, 2, 1, 0, 0 }, { 0x01a, 5, 0, 2, 0 },
    { 0x03a, 6, 0, 3, 0 }, { 0x004, 3, 4, 4, 0 }, { 0x03b, 6, 1, 20, 0 },
    { 0x00b, 4, 4, 22, 0 }, { 0x00c, 4, 5, 38, 0 }, { 0x01b, 5, 6, 70, 0 },
    { 0x01c, 5, 7, 134, 0 }, { 0x03c, 6, 7, 262, 0 }, { 0x07d, 7, 8, 390, 0 },
    { 0x03d, 6, 10, 646, 0 }, { 0x1ff, 9, 32, 1670, 0 }, { 0x001, 2, 0, OOB_VAL, 0 },
};
static const StdHuffLine HUFF_B9[] = {
    { 0x1fe, 9, 32, -32, 2 }, { 0x0fc, 8, 4, -31, 0 }, { 0x1fc, 9, 2, -15, 0 },
    { 0x0fd, 8, 2, -11, 0 }, { 0x1fd, 9, 1, -7, 0 }, { 0x07c, 7, 1, -5, 0 },
    { 0x00a, 4, 1, -3, 0 }, { 0x002, 3, 1, -1, 0 }, { 0x003, 3, 1, 1, 0 },
    { 0x01a, 5, 1, 3, 0 }, { 0x03a, 6, 1, 5, 0 }, { 0x004, 3, 5, 7, 0 },
    { 0x03b, 6, 2, 39, 0 }, { 0x00b, 4, 5, 43, 0 }, { 0x00c, 4, 6, 75, 0 },
    { 0x01b, 5, 7, 139, 0 }, { 0x01c, 5, 8, 267, 0 }, { 0x03c, 6, 8, 523, 0 },
    { 0x07d, 7, 9, 779, 0 }, { 0x03d, 6, 11, 1291, 0 }, { 0x1ff, 9, 32, 3339, 0 },
    { 0x000, 2, 0, OOB_VAL, 0 },
};
static const StdHuffLine HUFF_B10[] = {
    { 0x0fe, 8, 32, -22, 2 }, { 0x07a, 7, 4, -21, 0 }, { 0x0fc, 8, 0, -5, 0 },
    { 0x07b, 7, 0, -4, 0 }, { 0x018, 5, 0, -3, 0 }, { 0x000, 2, 2, -2, 0 },
    { 0x019, 5, 0, 2, 0 }, { 0x036, 6, 0, 3, 0 }, { 0x07c, 7, 0, 4, 0 },
    { 0x0fd, 8, 0, 5, 0 }, { 0x001, 2, 6, 6, 0 }, { 0x01a, 5, 5, 70, 0 },
    { 0x037, 6, 5, 102, 0 }, { 0x038, 6, 6, 134, 0 }, { 0x039, 6, 7, 198, 0 },
    { 0x03a, 6, 8, 326, 0 }, { 0x03b, 6, 9, 582, 0 }, { 0x03c, 6, 10, 1094, 0 },
    { 0x07d, 7, 11, 2118, 0 }, { 0x0ff, 8, 32, 4166, 0 }, { 0x002, 2, 0, OOB_VAL, 0 },
};
static const StdHuffLine HUFF_B11[] = {
    { 0x00, 1, 0, 1, 0 }, { 0x02, 2, 1, 2, 0 }, { 0x0c, 4, 0, 4, 0 },
    { 0x0d, 4, 1, 5, 0 }, { 0x1c, 5, 1, 7, 0 }, { 0x1d, 5, 2, 9, 0 },
    { 0x3c, 6, 2, 13, 0 }, { 0x7a, 7, 2, 17, 0 }, { 0x7b, 7, 3, 21, 0 },
    { 0x7c, 7, 4, 29, 0 }, { 0x7d, 7, 5, 45, 0 }, { 0x7e, 7, 6, 77, 0 },
    { 0x7f, 7, 32, 141, 0 },
};
static const StdHuffLine HUFF_B12[] = {
    { 0x00, 1, 0, 1, 0 }, { 0x02, 2, 0, 2, 0 }, { 0x06, 3, 1, 3, 0 },
    { 0x1c, 5, 0, 5, 0 }, { 0x1d, 6, 1, 6, 0 }, { 0x3c, 7, 1, 8, 0 },
    { 0x7a, 7, 0, 10, 0 }, { 0x7b, 7, 1, 11, 0 }, { 0x7c, 7, 2, 13, 0 },
    { 0x7d, 7, 3, 17, 0 }, { 0x7e, 7, 4, 25, 0 }, { 0xfe, 8, 5, 41, 0 },
    { 0xff, 8, 32, 73, 0 },
};
static const StdHuffLine HUFF_B13[] = {
    { 0x00, 1, 0, 1, 0 }, { 0x04, 3, 0, 2, 0 }, { 0x0c, 4, 0, 3, 0 },
    { 0x1c, 5, 0, 4, 0 }, { 0x0d, 4, 1, 5, 0 }, { 0x05, 3, 3, 7, 0 },
    { 0x3a, 6, 1, 15, 0 }, { 0x3b, 6, 2, 17, 0 }, { 0x3c, 6, 3, 21, 0 },
    { 0x3d, 6, 4, 29, 0 }, { 0x3e, 6, 5, 45, 0 }, { 0x7e, 7, 6, 77, 0 },
    { 0x7f, 7, 32, 141, 0 },
};
static const StdHuffLine HUFF_B14[] = {
    { 0x04, 3, 0, -2, 0 }, { 0x05, 3, 0, -1, 0 }, { 0x00, 1, 0, 0, 0 },
    { 0x06, 3, 0, 1, 0 }, { 0x07, 3, 0, 2, 0 },
};
static const StdHuffLine HUFF_B15[] = {
    { 0x7e, 7, 32, -25, 2 }, { 0x7c, 7, 4, -24, 0 }, { 0x3c, 6, 2, -8, 0 },
    { 0x1c, 5, 1, -4, 0 }, { 0x0c, 4, 0, -2, 0 }, { 0x04, 3, 0, -1, 0 },
    { 0x00, 1, 0, 0, 0 }, { 0x05, 3, 0, 1, 0 }, { 0x0d, 4, 0, 2, 0 },
    { 0x1d, 5, 1, 3, 0 }, { 0x3d, 6, 2, 5, 0 }, { 0x7d, 7, 4, 9, 0 },
    { 0x7f, 7, 32, 25, 0 },
};

// Encodes `val` using a standard Huffman table (Annex B.5), porting the
// reference encoder's algorithm (ITU-T T.88 JBIG2 Sample Software,
// Jb2_T4T6Lapper.cpp: JBIG2_HuffEnc). Rows must be in increasing `val`
// order (matching the tables above) except the out-of-band line, which
// this always places last.
static void huff_encode(BitWriter &bw, StdHuffTable t, int32_t val)
{
    const StdHuffLine *table = t.rows;
    size_t n = t.n;
    bool has_oob = (table[n - 1].val == OOB_VAL);
    if (val == OOB_VAL) {
        if (!has_oob) {
            fprintf(stderr, "huff_encode: OOB requested from a table without an OOB line\n");
            exit(1);
        }
        bw_put_bits(bw, (uint32_t)table[n - 1].code, table[n - 1].code_len);
        return;
    }
    size_t limit = n - (has_oob ? 1 : 0);

    // No standard table spans the whole integer range: B.4 and B.11-B.13
    // start at 1, B.14 covers only [-2,2]. An out-of-range value would
    // still produce output -- the search below clamps to the first or last
    // line -- but with an offset that doesn't round-trip, and the offset
    // field is often too narrow to carry the discrepancy anyway. Encoding
    // 0 with B.11 writes exactly the bits for 1: no diagnostic, and a
    // decoder silently reads back a different number than was intended.
    // The caller chooses both the table and the value, so a mismatch is a
    // bug here rather than a legal input; fail loudly instead of emitting
    // a stream that misreports itself.
    const StdHuffLine &last = table[limit - 1];
    bool below = table[0].val_func != 2 && val < table[0].val;   // val_func 2 = lower range line, open below
    bool above = last.range_bits < 32 &&
                 val > last.val + (int32_t)((1u << last.range_bits) - 1);
    if (below || above) {
        fprintf(stderr, "huff_encode: value %d not representable by the selected table "
                        "(lines %d..%d)\n", val, table[0].val, last.val);
        exit(1);
    }

    size_t k = 1;
    while (k < limit && val >= table[k].val)
        k++;
    k--;
    int32_t offset = (table[k].val_func == 2) ? (table[k].val - val) : (val - table[k].val);
    bw_put_bits(bw, (uint32_t)table[k].code, table[k].code_len);
    bw_put_bits(bw, (uint32_t)offset, table[k].range_bits);
}

// CCITT T.4/T.6 run-length codes: white/black terminating codes (run 0-63)
// and makeup codes (run 64, 128, ..., 2560), plus the T.6 2D mode control
// codes (pass/horizontal/vertical). Transcribed from the ITU-T T.88
// reference encoder (JBIG2 Sample Software, T4T6codec.h), reproduced here
// under its BSD-style ITU/ISO reference-software license.
struct RunCode {
    uint32_t run;
    uint32_t code;
    uint32_t code_len;
};

// Table layout, which mmr_write_run()'s index arithmetic depends on:
// entries [0,63] are terminating codes for a run of exactly that length,
// entries [64,103] are makeup codes for runs of 64, 128, ..., 2560. So a
// makeup run R sits at index R/64 + 63, and the largest makeup code (2560)
// is the repeatable one used to walk long runs down. mmr_check_tables()
// below verifies the data actually matches this.
static const int MMR_NTERM = 64;             // count of terminating codes, and the makeup step
static const int MMR_MAX_MAKEUP = 2560;
static const int MMR_MAX_MAKEUP_IDX = 103;   // == MMR_MAX_MAKEUP / MMR_NTERM + 63

static const RunCode MMR_WHITE_CODES[104] = {
    { 0, 0x35, 8 }, { 1, 0x07, 6 }, { 2, 0x07, 4 }, { 3, 0x08, 4 },
    { 4, 0x0B, 4 }, { 5, 0x0C, 4 }, { 6, 0x0E, 4 }, { 7, 0x0F, 4 },
    { 8, 0x13, 5 }, { 9, 0x14, 5 }, { 10, 0x07, 5 }, { 11, 0x08, 5 },
    { 12, 0x08, 6 }, { 13, 0x03, 6 }, { 14, 0x34, 6 }, { 15, 0x35, 6 },
    { 16, 0x2A, 6 }, { 17, 0x2B, 6 }, { 18, 0x27, 7 }, { 19, 0x0C, 7 },
    { 20, 0x08, 7 }, { 21, 0x17, 7 }, { 22, 0x03, 7 }, { 23, 0x04, 7 },
    { 24, 0x28, 7 }, { 25, 0x2B, 7 }, { 26, 0x13, 7 }, { 27, 0x24, 7 },
    { 28, 0x18, 7 }, { 29, 0x02, 8 }, { 30, 0x03, 8 }, { 31, 0x1A, 8 },
    { 32, 0x1B, 8 }, { 33, 0x12, 8 }, { 34, 0x13, 8 }, { 35, 0x14, 8 },
    { 36, 0x15, 8 }, { 37, 0x16, 8 }, { 38, 0x17, 8 }, { 39, 0x28, 8 },
    { 40, 0x29, 8 }, { 41, 0x2A, 8 }, { 42, 0x2B, 8 }, { 43, 0x2C, 8 },
    { 44, 0x2D, 8 }, { 45, 0x04, 8 }, { 46, 0x05, 8 }, { 47, 0x0A, 8 },
    { 48, 0x0B, 8 }, { 49, 0x52, 8 }, { 50, 0x53, 8 }, { 51, 0x54, 8 },
    { 52, 0x55, 8 }, { 53, 0x24, 8 }, { 54, 0x25, 8 }, { 55, 0x58, 8 },
    { 56, 0x59, 8 }, { 57, 0x5A, 8 }, { 58, 0x5B, 8 }, { 59, 0x4A, 8 },
    { 60, 0x4B, 8 }, { 61, 0x32, 8 }, { 62, 0x33, 8 }, { 63, 0x34, 8 },
    { 64, 0x1B, 5 }, { 128, 0x12, 5 }, { 192, 0x17, 6 }, { 256, 0x37, 7 },
    { 320, 0x36, 8 }, { 384, 0x37, 8 }, { 448, 0x64, 8 }, { 512, 0x65, 8 },
    { 576, 0x68, 8 }, { 640, 0x67, 8 }, { 704, 0xCC, 9 }, { 768, 0xCD, 9 },
    { 832, 0xD2, 9 }, { 896, 0xD3, 9 }, { 960, 0xD4, 9 }, { 1024, 0xD5, 9 },
    { 1088, 0xD6, 9 }, { 1152, 0xD7, 9 }, { 1216, 0xD8, 9 }, { 1280, 0xD9, 9 },
    { 1344, 0xDA, 9 }, { 1408, 0xDB, 9 }, { 1472, 0x98, 9 }, { 1536, 0x99, 9 },
    { 1600, 0x9A, 9 }, { 1664, 0x18, 6 }, { 1728, 0x9B, 9 }, { 1792, 0x08, 12 },
    { 1856, 0x0C, 11 }, { 1920, 0x0D, 11 }, { 1984, 0x12, 11 }, { 2048, 0x13, 12 },
    { 2112, 0x14, 12 }, { 2176, 0x15, 12 }, { 2240, 0x16, 12 }, { 2304, 0x17, 12 },
    { 2368, 0x1C, 12 }, { 2432, 0x1D, 12 }, { 2496, 0x1E, 12 }, { 2560, 0x1F, 12 },
};
static const RunCode MMR_BLACK_CODES[104] = {
    { 0, 0x37, 10 }, { 1, 0x02, 3 }, { 2, 0x03, 2 }, { 3, 0x02, 2 },
    { 4, 0x03, 3 }, { 5, 0x03, 4 }, { 6, 0x02, 4 }, { 7, 0x03, 5 },
    { 8, 0x05, 6 }, { 9, 0x04, 6 }, { 10, 0x04, 7 }, { 11, 0x05, 7 },
    { 12, 0x07, 7 }, { 13, 0x04, 8 }, { 14, 0x07, 8 }, { 15, 0x18, 9 },
    { 16, 0x17, 10 }, { 17, 0x18, 10 }, { 18, 0x08, 10 }, { 19, 0x67, 11 },
    { 20, 0x68, 11 }, { 21, 0x6C, 11 }, { 22, 0x37, 11 }, { 23, 0x28, 11 },
    { 24, 0x17, 11 }, { 25, 0x18, 11 }, { 26, 0xCA, 12 }, { 27, 0xCB, 12 },
    { 28, 0xCC, 12 }, { 29, 0xCD, 12 }, { 30, 0x68, 12 }, { 31, 0x69, 12 },
    { 32, 0x6A, 12 }, { 33, 0x6B, 12 }, { 34, 0xD2, 12 }, { 35, 0xD3, 12 },
    { 36, 0xD4, 12 }, { 37, 0xD5, 12 }, { 38, 0xD6, 12 }, { 39, 0xD7, 12 },
    { 40, 0x6C, 12 }, { 41, 0x6D, 12 }, { 42, 0xDA, 12 }, { 43, 0xDB, 12 },
    { 44, 0x54, 12 }, { 45, 0x55, 12 }, { 46, 0x56, 12 }, { 47, 0x57, 12 },
    { 48, 0x64, 12 }, { 49, 0x65, 12 }, { 50, 0x52, 12 }, { 51, 0x53, 12 },
    { 52, 0x24, 12 }, { 53, 0x37, 12 }, { 54, 0x38, 12 }, { 55, 0x27, 12 },
    { 56, 0x28, 12 }, { 57, 0x58, 12 }, { 58, 0x59, 12 }, { 59, 0x2B, 12 },
    { 60, 0x2C, 12 }, { 61, 0x5A, 12 }, { 62, 0x66, 12 }, { 63, 0x67, 12 },
    { 64, 0x0F, 10 }, { 128, 0xC8, 12 }, { 192, 0xC9, 12 }, { 256, 0x5B, 12 },
    { 320, 0x33, 12 }, { 384, 0x34, 12 }, { 448, 0x35, 12 }, { 512, 0x6C, 13 },
    { 576, 0x6D, 13 }, { 640, 0x4A, 13 }, { 704, 0x4B, 13 }, { 768, 0x4C, 13 },
    { 832, 0x4D, 13 }, { 896, 0x72, 13 }, { 960, 0x73, 13 }, { 1024, 0x74, 13 },
    { 1088, 0x75, 13 }, { 1152, 0x76, 13 }, { 1216, 0x77, 13 }, { 1280, 0x52, 13 },
    { 1344, 0x53, 13 }, { 1408, 0x54, 13 }, { 1472, 0x55, 13 }, { 1536, 0x5A, 13 },
    { 1600, 0x5B, 13 }, { 1664, 0x64, 13 }, { 1728, 0x65, 13 }, { 1792, 0x08, 13 },
    { 1856, 0x0C, 13 }, { 1920, 0x0D, 13 }, { 1984, 0x12, 13 }, { 2048, 0x13, 13 },
    { 2112, 0x14, 13 }, { 2176, 0x15, 13 }, { 2240, 0x16, 13 }, { 2304, 0x17, 13 },
    { 2368, 0x1C, 13 }, { 2432, 0x1D, 13 }, { 2496, 0x1E, 13 }, { 2560, 0x1F, 13 },
};

// Encodes one run of `length` pixels of colour `black` using CCITT T.4
// Modified Huffman run codes: repeated 2560-run makeup codes for
// length > 2560, one makeup code for length >= 64, then exactly one
// terminating code (0-63). Port of codsub.cpp: Cmh.
// Checks the run tables against the index arithmetic mmr_write_run() uses.
// The `run` field is read nowhere else, so without this it is a comment
// that the compiler cannot keep honest: re-transcribing a row, reordering
// the makeup block, or extending either table would silently change which
// code a given run length maps to, and the output would still be
// well-formed MMR -- just decoding to different run lengths than intended.
static void mmr_check_tables(void)
{
    const RunCode *tables[2] = { MMR_WHITE_CODES, MMR_BLACK_CODES };
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i <= MMR_MAX_MAKEUP_IDX; i++) {
            uint32_t expect = (i < MMR_NTERM) ? (uint32_t)i
                                              : (uint32_t)(i - (MMR_NTERM - 1)) * MMR_NTERM;
            if (tables[t][i].run != expect) {
                fprintf(stderr, "mmr_check_tables: %s table entry %d codes run %u, expected %u\n",
                        t ? "black" : "white", i, tables[t][i].run, expect);
                exit(1);
            }
        }
    }
}

static void mmr_write_run(BitWriter &bw, int length, bool black)
{
    // Callers derive `length` from changing-element differences (a1-a0,
    // a2-a1), which are non-negative by construction -- but nothing in
    // this function establishes that, and a negative length clears both
    // guards below to reach table[length % MMR_NTERM], a negative index.
    // That is an out-of-bounds read, not merely a wrong code, so it is the
    // one precondition here worth checking rather than assuming.
    if (length < 0) {
        fprintf(stderr, "mmr_write_run: negative run length %d\n", length);
        exit(1);
    }
    const RunCode *table = black ? MMR_BLACK_CODES : MMR_WHITE_CODES;
    while (length > MMR_MAX_MAKEUP) {
        bw_put_bits(bw, table[MMR_MAX_MAKEUP_IDX].code, table[MMR_MAX_MAKEUP_IDX].code_len);
        length -= MMR_MAX_MAKEUP;
    }
    if (length >= MMR_NTERM) {
        // In [64, 103]: the guard above bounds length below, the loop above bounds it by 2560.
        int idx = length / MMR_NTERM + (MMR_NTERM - 1);
        bw_put_bits(bw, table[idx].code, table[idx].code_len);
    }
    int term = length % MMR_NTERM;
    bw_put_bits(bw, table[term].code, table[term].code_len);
}

// Finds the next T.6 "changing element" on `buf` (length `width`) at or
// after `sptr` whose colour is `target`: skips the run already matching
// `target` (a no-op if buf[sptr] doesn't match it), then skips the
// following run of the opposite colour, landing on the transition back to
// `target`. sptr == -1 is the imaginary pixel before the line, always
// white (0). Port of codsub.cpp: Cdetchg.
static int mmr_find_change(int width, int sptr, int target, const uint8_t *buf)
{
    int p = sptr;
    for (;;) {
        int c = (p < 0) ? 0 : buf[p];
        if (c != target)
            break;
        p++;
        if (p >= width)
            return width;
    }
    for (;;) {
        int c = (p < 0) ? 0 : buf[p];
        if (c == target)
            break;
        p++;
        if (p >= width)
            return width;
    }
    return p;
}

// Computes the T.6 a1/a2 (on the coding line `lbuf`) and b1/b2 (on the
// reference line `rbuf`) changing elements relative to a0, leaving
// already-known (non-negative) values untouched. Port of codsub.cpp:
// Cdetab.
static void mmr_find_ab(int width, int a0, int &a1, int &a2, int &b1, int &b2,
                         const uint8_t *lbuf, const uint8_t *rbuf)
{
    int target = (a0 >= 0) ? (1 ^ lbuf[a0]) : 1;
    if (a1 < 0)
        a1 = mmr_find_change(width, a0, target, lbuf);
    if (b1 < 0)
        b1 = mmr_find_change(width, a0, target, rbuf);

    target ^= 1;
    if (a1 >= width)
        a2 = width;
    else if (a2 < 0)
        a2 = mmr_find_change(width, a1, target, lbuf);
    if (b1 >= width)
        b2 = width;
    else if (b2 < 0)
        b2 = mmr_find_change(width, b1, target, rbuf);
}

// Encodes one row via T.6 2D coding (pass/horizontal/vertical modes)
// against the previous row `rbuf` as reference. Port of codsub.cpp: Cdim2.
static void mmr_encode_line(BitWriter &bw, int width, const uint8_t *lbuf, const uint8_t *rbuf)
{
    int a0 = -1, a1 = -1, a2 = -1, b1 = -1, b2 = -1;
    while (a0 < width) {
        mmr_find_ab(width, a0, a1, a2, b1, b2, lbuf, rbuf);
        if (a1 > b2) {
            bw_put_bits(bw, 0x1, 4);   // PASS
            a0 = b2;
            b1 = b2 = -1;
        } else if (std::abs(a1 - b1) <= 3) {
            switch (a1 - b1 + 3) {
            case 0: bw_put_bits(bw, 0x2, 7); break;   // VL3
            case 1: bw_put_bits(bw, 0x2, 6); break;   // VL2
            case 2: bw_put_bits(bw, 0x2, 3); break;   // VL1
            case 3: bw_put_bits(bw, 0x1, 1); break;   // V0
            case 4: bw_put_bits(bw, 0x3, 3); break;   // VR1
            case 5: bw_put_bits(bw, 0x3, 6); break;   // VR2
            case 6: bw_put_bits(bw, 0x3, 7); break;   // VR3
            }
            a0 = a1;
            if (b2 > a1) {
                if (b1 <= a1)
                    b1 = b2;
                else if (rbuf[a1] != lbuf[a1])
                    b1 = b2;
                else
                    b1 = -1;
            } else {
                b1 = -1;
            }
            a1 = a2;
            a2 = b2 = -1;
        } else {
            bw_put_bits(bw, 0x1, 3);   // HORZ
            int worb, length;
            if (a0 >= 0) {
                worb = lbuf[a0];
                length = a1 - a0;
            } else {
                worb = 0;
                length = a1 - a0 - 1;
            }
            mmr_write_run(bw, length, worb != 0);
            worb ^= 1;
            length = a2 - a1;
            mmr_write_run(bw, length, worb != 0);
            a0 = a2;
            if (a2 >= b1)
                b1 = b2 = -1;
            a1 = a2 = -1;
        }
    }
}

// Encodes a width x height 1bpp bitmap (row-major, one byte per pixel, 0 or
// 1) as T.6 (MMR) 2D-coded data, without a trailing EOFB -- valid whenever
// the decoder knows the byte count in advance (6.2.6), which holds for
// every JBIG2 use of MMR (pattern dictionaries, symbol dictionary
// collective bitmaps, and known-length generic regions). Ported from the
// ITU-T T.88 reference encoder (JBIG2 Sample Software, codsub.cpp:
// Cdim2/Cdetchg/Cdetab/Cmh), reproduced here under its BSD-style ITU/ISO
// reference-software license.
static std::vector<uint8_t> mmr_encode(int width, int height, const uint8_t *pixels)
{
    BitWriter bw;
    std::vector<uint8_t> refline((size_t)width, 0);   // reference line starts all-white
    for (int y = 0; y < height; y++) {
        const uint8_t *cur = pixels + (size_t)y * width;
        mmr_encode_line(bw, width, cur, refline.data());
        refline.assign(cur, cur + width);
    }
    bw_finish(bw);
    return bw.bytes;
}

// MQ arithmetic coder (Annex E) probability-estimation state table: for
// each of the 47 states, the LPS/MPS-switch next-state indices and the Qe
// probability estimate, packed as one uint32 per row (switch:8,
// nmps:8, nlps:8, Qe:16 -- read out via shifts/masks below, matching the
// reference layout). Transcribed from the ITU-T T.88 reference encoder
// (JBIG2 Sample Software, MQ_codec.h: QeIndexTable), reproduced here under
// its BSD-style ITU/ISO reference-software license.
static const uint32_t MQ_QE_TABLE[47] = {
    0x81015601, 0x06023401, 0x09031801, 0x0C040AC1, 0x1D050521, 0x21260221,
    0x86075601, 0x0E085401, 0x0E094801, 0x0E0A3801, 0x110B3001, 0x120C2401,
    0x140D1C01, 0x151D1601, 0x8E0F5601, 0x0E105401, 0x0F115101, 0x10124801,
    0x11133801, 0x12143401, 0x13153001, 0x13162801, 0x14172401, 0x15182201,
    0x16191C01, 0x171A1801, 0x181B1601, 0x191C1401, 0x1A1D1201, 0x1B1E1101,
    0x1C1F0AC1, 0x1D2009C1, 0x1E2108A1, 0x1F220521, 0x20230441, 0x212402A1,
    0x22250221, 0x23260141, 0x24270111, 0x25280085, 0x26290049, 0x272A0025,
    0x282B0015, 0x292C0009, 0x2A2D0005, 0x2B2D0001, 0x2E2E5601,
};

// MQ arithmetic encoder state. Port of MQ_codec.cpp (InitMQ_Codec/Enc_MQ/
// MQ_ByteOut/MQ_flush), reproduced under the same reference-software
// license noted above. Each encode call needs its own context-state array
// (one entry per possible CONTEXT value, zero-initialized -- Annex E.3.7
// resets these to zero at the start of every segment).
struct MQEncoder {
    uint32_t Creg = 0;
    int32_t Areg = 0x8000;
    int ctreg = 12;
    uint8_t B_buf = 0;
    bool first = true;
    std::vector<uint8_t> out;
};

static void mq_byte_out(MQEncoder &e)
{
    if (e.first) {
        e.B_buf = (uint8_t)(e.Creg >> 19);
        e.Creg &= 0x7ffff;
        e.ctreg = 8;
        e.first = false;
        return;
    }
    bool ff_flag;
    if (e.B_buf == 0xff) {
        ff_flag = true;
    } else {
        e.B_buf = (uint8_t)(e.B_buf + ((e.Creg & 0x8000000) ? 1 : 0));
        e.Creg &= 0x7ffffff;
        ff_flag = (e.B_buf == 0xff);
    }
    e.out.push_back(e.B_buf);
    if (ff_flag) {
        e.B_buf = (uint8_t)(e.Creg >> 20);
        e.Creg &= 0xfffff;
        e.ctreg = 7;
    } else {
        e.B_buf = (uint8_t)(e.Creg >> 19);
        e.Creg &= 0x7ffff;
        e.ctreg = 8;
    }
}

// Encodes one binary decision `d` (0 or 1) under context `cx`, whose
// adaptive state lives in `cx_state[cx]` (bits 0-6: state index into
// MQ_QE_TABLE; bit 7: current MPS). Port of MQ_codec.cpp: Enc_MQ.
static void mq_encode_bit(MQEncoder &e, std::vector<uint8_t> &cx_state, uint32_t cx, int d)
{
    uint8_t index = cx_state[cx] & 0x7f;
    uint8_t mps = cx_state[cx] & 0x80;
    uint32_t packed = MQ_QE_TABLE[index];
    int32_t qe = (int32_t)(packed & 0xffff);

    e.Areg -= qe;
    uint8_t di = (uint8_t)(d << 7);

    if (di ^ mps) {
        // LPS
        if (e.Areg < qe)
            e.Creg += (uint32_t)qe;
        else
            e.Areg = qe;
        cx_state[cx] = (uint8_t)((packed >> 24) ^ mps);
    } else {
        // MPS
        if (e.Areg < qe) {
            e.Areg = qe;
            cx_state[cx] = (uint8_t)(((packed >> 16) & 0xff) | mps);
        } else {
            e.Creg += (uint32_t)qe;
            if (!(e.Areg & 0x8000))
                cx_state[cx] = (uint8_t)(((packed >> 16) & 0xff) | mps);
        }
    }

    for (; !(e.Areg & 0x8000); e.Areg <<= 1) {
        e.Creg <<= 1;
        e.ctreg--;
        if (e.ctreg == 0)
            mq_byte_out(e);
    }
}

// Terminates the arithmetic codeword. Port of MQ_codec.cpp: MQ_setbits/MQ_flush.
static void mq_flush(MQEncoder &e)
{
    uint32_t tmp = e.Creg + (uint32_t)e.Areg;
    e.Creg |= 0xffff;
    if (e.Creg >= tmp)
        e.Creg -= 0x8000;

    e.Creg <<= e.ctreg;
    mq_byte_out(e);
    e.Creg <<= e.ctreg;
    mq_byte_out(e);
    if (e.B_buf != 0xff) {
        e.Creg <<= e.ctreg;
        mq_byte_out(e);
    }
    e.Creg <<= e.ctreg;
    mq_byte_out(e);
}

// Annex A.2/A.3: arithmetic integer decoding procedure -- used (as IADH,
// IADW, IAEX, IAAI, IADT, IAFS, IADS, IAIT, IARI, IARDW, IARDH, IARDX,
// IARDY, each its own independently-adapting instance of this same
// algorithm) throughout arithmetic-coded symbol dictionaries and text
// regions, the one piece of entropy-coding infrastructure this generator
// was missing to produce real content for either. One 512-entry adaptive
// context array per named procedure, indexed by a running PREV value that
// doubles as both the context selector and (via CJBig2_ArithIntDecoder::
// Decode()'s exact PREV update sequence, jbig2_arith_int_decoder.cpp,
// which this mirrors bit-for-bit) the accumulator for the prefix code
// that selects how many magnitude bits follow.
struct ArithIntCtx {
    std::vector<uint8_t> cx;
    ArithIntCtx() : cx(512, 0) {}
};

struct ArithIntBucket {
    int need_bits;
    int32_t value_offset;
};
// kArithIntDecodeData in jbig2_arith_int_decoder.cpp, verbatim: bucket b's
// prefix is b ones followed by a zero (or, for b == 5, five ones with no
// terminating zero -- RecursiveDecode() stops at depth 5 unconditionally),
// followed by need_bits magnitude bits, giving value = value_offset + that
// magnitude.
static const ArithIntBucket ARITH_INT_BUCKETS[6] = {
    { 2, 0 }, { 4, 4 }, { 6, 20 }, { 8, 84 }, { 12, 340 }, { 32, 4436 },
};

// Encodes `value` (magnitude and sign) via the procedure above; `oob`
// (out-of-band, only meaningful where the caller's field allows it, e.g.
// IADW/IADS marking the end of a height class/strip) encodes the same
// sign-negative-zero sentinel CJBig2_ArithIntDecoder::Decode() reports by
// returning false for -- S == 1 (negative) with a zero magnitude, which a
// genuine value of 0 (S == 0) never produces, so the two never collide.
static void mq_encode_arith_int(MQEncoder &mq, ArithIntCtx &ictx, int32_t value, bool oob = false)
{
    int S = oob ? 1 : (value < 0 ? 1 : 0);
    uint32_t mag = oob ? 0 : (uint32_t)(value < 0 ? -(int64_t)value : (int64_t)value);

    int bucket = 5;
    for (int b = 0; b < 5; b++) {
        if (mag < (uint32_t)ARITH_INT_BUCKETS[b + 1].value_offset) {
            bucket = b;
            break;
        }
    }
    uint32_t nTemp = mag - (uint32_t)ARITH_INT_BUCKETS[bucket].value_offset;

    int PREV = 1;
    mq_encode_bit(mq, ictx.cx, (uint32_t)PREV, S);
    PREV = (PREV << 1) | S;
    for (int i = 0; i < bucket; i++) {
        mq_encode_bit(mq, ictx.cx, (uint32_t)PREV, 1);
        PREV = (PREV << 1) | 1;
    }
    if (bucket < 5) {
        mq_encode_bit(mq, ictx.cx, (uint32_t)PREV, 0);
        PREV = (PREV << 1) | 0;
    }
    int need_bits = ARITH_INT_BUCKETS[bucket].need_bits;
    for (int i = need_bits - 1; i >= 0; i--) {
        int bit = (int)((nTemp >> i) & 1u);
        mq_encode_bit(mq, ictx.cx, (uint32_t)PREV, bit);
        PREV = (PREV << 1) | bit;
        if (PREV >= 256)
            PREV = (PREV & 511) | 256;
    }
}

// Annex A.3: arithmetic symbol ID decoding procedure (IAID) -- just
// SBSYMCODELEN raw bits gathered MSB-first through their own adaptive
// PREV-indexed context tree (CJBig2_ArithIaidDecoder::Decode(),
// jbig2_arith_int_decoder.cpp), unlike IAx's variable-length prefix code
// above. `cx` must be sized 1u << sbsymcodelen, matching a decoder's own
// iaid_.resize(1 << SBSYMCODELEN).
static void mq_encode_arith_iaid(MQEncoder &mq, std::vector<uint8_t> &cx, int sbsymcodelen, uint32_t value)
{
    int PREV = 1;
    for (int i = sbsymcodelen - 1; i >= 0; i--) {
        int bit = (int)((value >> i) & 1u);
        mq_encode_bit(mq, cx, (uint32_t)PREV, bit);
        PREV = (PREV << 1) | bit;
    }
}

// 6.2.5.7 step 3b: the fixed context each GBTEMPLATE reuses to code SLTP
// when TPGDON is on. These are not spec-mandated numbers -- Figure 8-11
// fix the *pixel pattern*, but "the order of this gathering is not
// standardized" (6.2.5.4) means the resulting context number depends on
// each implementation's own bit-gathering order. Values here are pdfium's
// own literal constants (jbig2_grd_proc.cpp), not the ITU reference
// encoder's (Jb2_MQLapper.cpp: 0xc395/0x0795/0x0271/0x02c5, which agree
// only for template 1) -- since our bit layouts above were matched to
// pdfium's, the SLTP context has to be pdfium's too, or the two sides
// disagree on which adaptive-state slot even under a spec-conformant
// stream.
static const uint32_t GD_TPGDON_CX[4] = { 0x9b25, 0x0795, 0x00e5, 0x0195 };

// Encodes a width x height 1bpp bitmap (row-major, one byte per pixel, 0 or
// 1) as arithmetic-coded generic region data using GBTEMPLATE 0 (the
// 16-bit, 4-AT-pixel template). The context bit layout (bits 0-3 = 4
// causal pixels on the current row; bit 4 = AT1; bits 5-9 = 5 pixels from
// the row above; bits 10-11 = AT2/AT3; bits 12-14 = 3 pixels from two rows
// above; bit 15 = AT4) was reverse-engineered from jbig2dec's
// jbig2_decode_generic_template0_unopt(), since 6.2.5.7 leaves the
// gathering order implementation-defined -- interoperating with a real
// decoder means matching its specific choice, not just any consistent one.
// `pixels` is mutated in place when tpgdon is set: 6.2.5.7 step 3c requires
// a row coded as "typical" (LTP=1) to be identical to the row above, so any
// row this function decides is typical gets overwritten with that row's
// values before both encoding and return, leaving `pixels` holding exactly
// what a decoder reconstructs.
// Core of mq_encode_generic_template0(), operating on a caller-owned
// MQEncoder/context array instead of a fresh one -- lets a caller encode
// several regions back to back into one continuous arithmetic-coded
// stream sharing adaptive state throughout, the same way
// CJBig2_HTRDProc::DecodeArith reuses one CJBig2_ArithDecoder and one
// gbContexts span across every one of a halftone region's bitplanes
// (jbig2_htrd_proc.cpp) rather than restarting per plane. Every other real
// generic-region content (a standalone generic region segment, or a
// pattern dictionary's collective bitmap) is exactly one such region, so
// mq_encode_generic_template0() below is this with a private one-shot
// MQEncoder/context array and its own mq_flush().
// `skip` (nullptr, or one byte per pixel, row-major) mirrors USESKIP/SKIP
// (6.2.5.7 step 3(c)(x)): a decoder never calls Decode() for a skipped
// pixel at all, just takes it as 0, so this must not spend a bit on it
// either -- forcing `pixels` to 0 there too, matching what a decoder's own
// zero-initialized image already holds when SetPixel() is never called
// for that position.
static void mq_encode_generic_template0_into(MQEncoder &mq, std::vector<uint8_t> &cx_state,
                                               int width, int height, uint8_t *pixels,
                                               int atx1, int aty1, int atx2, int aty2,
                                               int atx3, int aty3, int atx4, int aty4,
                                               bool tpgdon, const uint8_t *skip = nullptr)
{
    auto px = [&](int x, int y) -> uint32_t {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return pixels[(size_t)y * width + x];
    };

    int ltp = 0;
    for (int y = 0; y < height; y++) {
        if (tpgdon) {
            int want = (int)(urand() & 1);
            mq_encode_bit(mq, cx_state, GD_TPGDON_CX[0], want ^ ltp);
            ltp = want;
        }
        if (ltp) {
            for (int x = 0; x < width; x++)
                pixels[(size_t)y * width + x] = (uint8_t)px(x, y - 1);
            continue;
        }
        for (int x = 0; x < width; x++) {
            if (skip && skip[(size_t)y * width + x]) {
                pixels[(size_t)y * width + x] = 0;
                continue;
            }
            uint32_t cx = 0;
            cx |= px(x - 1, y) << 0;
            cx |= px(x - 2, y) << 1;
            cx |= px(x - 3, y) << 2;
            cx |= px(x - 4, y) << 3;
            cx |= px(x + atx1, y + aty1) << 4;
            cx |= px(x + 2, y - 1) << 5;
            cx |= px(x + 1, y - 1) << 6;
            cx |= px(x + 0, y - 1) << 7;
            cx |= px(x - 1, y - 1) << 8;
            cx |= px(x - 2, y - 1) << 9;
            cx |= px(x + atx2, y + aty2) << 10;
            cx |= px(x + atx3, y + aty3) << 11;
            cx |= px(x + 1, y - 2) << 12;
            cx |= px(x + 0, y - 2) << 13;
            cx |= px(x - 1, y - 2) << 14;
            cx |= px(x + atx4, y + aty4) << 15;
            int bit = pixels[(size_t)y * width + x] ? 1 : 0;
            mq_encode_bit(mq, cx_state, cx, bit);
        }
    }
}

static std::vector<uint8_t> mq_encode_generic_template0(int width, int height, uint8_t *pixels,
                                                          int atx1, int aty1, int atx2, int aty2,
                                                          int atx3, int aty3, int atx4, int aty4,
                                                          bool tpgdon)
{
    MQEncoder mq;
    std::vector<uint8_t> cx_state(1u << 16, 0);
    mq_encode_generic_template0_into(mq, cx_state, width, height, pixels,
                                      atx1, aty1, atx2, aty2, atx3, aty3, atx4, aty4, tpgdon);
    mq_flush(mq);
    return mq.out;
}

// Encodes a width x height 1bpp bitmap as arithmetic-coded generic region
// data using GBTEMPLATE 1, 2, or 3. Unlike template 0, these three take
// exactly one AT pixel (7.4.6.3), and their contexts are 13, 10 and 10
// bits wide respectively -- matching the 8192/1024/1024 context-array
// sizes a decoder allocates for them.
//
// As with template 0, 6.2.5.7 leaves the bit-gathering order
// implementation-defined, so these layouts were recovered from pdfium's
// CJBig2_GRDProc::ProgressiveDecodeArithTemplate{1,2,3}Unopt by unrolling
// their sliding-window accumulators (val_prev2/val_prev1/val_current, each
// masked to a fixed width and fed one new pixel per column) back into
// per-pixel coordinates. Each accumulator's mask width is exactly the
// number of context bits it contributes, and its newest pixel is the one
// the update reads, which pins both the span and the bit order.
// `pixels` is mutated in place when tpgdon is set -- see
// mq_encode_generic_template0()'s comment on the same parameter.
// Core of mq_encode_generic_template123(), operating on a caller-owned
// MQEncoder/context array -- see mq_encode_generic_template0_into()'s
// comment for why (halftone region bitplanes).
// `skip`: see mq_encode_generic_template0_into()'s comment on the same
// parameter.
static void mq_encode_generic_template123_into(MQEncoder &mq, std::vector<uint8_t> &cx_state,
                                                 int width, int height,
                                                 uint8_t *pixels, int gbtemplate,
                                                 int atx1, int aty1, bool tpgdon,
                                                 const uint8_t *skip = nullptr)
{
    auto px = [&](int x, int y) -> uint32_t {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return pixels[(size_t)y * width + x];
    };

    int ltp = 0;
    for (int y = 0; y < height; y++) {
        if (tpgdon) {
            int want = (int)(urand() & 1);
            mq_encode_bit(mq, cx_state, GD_TPGDON_CX[gbtemplate], want ^ ltp);
            ltp = want;
        }
        if (ltp) {
            for (int x = 0; x < width; x++)
                pixels[(size_t)y * width + x] = (uint8_t)px(x, y - 1);
            continue;
        }
        for (int x = 0; x < width; x++) {
            if (skip && skip[(size_t)y * width + x]) {
                pixels[(size_t)y * width + x] = 0;
                continue;
            }
            uint32_t cx = 0;
            if (gbtemplate == 1) {
                cx |= px(x - 1, y) << 0;
                cx |= px(x - 2, y) << 1;
                cx |= px(x - 3, y) << 2;
                cx |= px(x + atx1, y + aty1) << 3;
                cx |= px(x + 2, y - 1) << 4;
                cx |= px(x + 1, y - 1) << 5;
                cx |= px(x + 0, y - 1) << 6;
                cx |= px(x - 1, y - 1) << 7;
                cx |= px(x - 2, y - 1) << 8;
                cx |= px(x + 2, y - 2) << 9;
                cx |= px(x + 1, y - 2) << 10;
                cx |= px(x + 0, y - 2) << 11;
                cx |= px(x - 1, y - 2) << 12;
            } else if (gbtemplate == 2) {
                cx |= px(x - 1, y) << 0;
                cx |= px(x - 2, y) << 1;
                cx |= px(x + atx1, y + aty1) << 2;
                cx |= px(x + 1, y - 1) << 3;
                cx |= px(x + 0, y - 1) << 4;
                cx |= px(x - 1, y - 1) << 5;
                cx |= px(x - 2, y - 1) << 6;
                cx |= px(x + 1, y - 2) << 7;
                cx |= px(x + 0, y - 2) << 8;
                cx |= px(x - 1, y - 2) << 9;
            } else {
                // Template 3 samples only one previous row (6.2.5.3).
                cx |= px(x - 1, y) << 0;
                cx |= px(x - 2, y) << 1;
                cx |= px(x - 3, y) << 2;
                cx |= px(x - 4, y) << 3;
                cx |= px(x + atx1, y + aty1) << 4;
                cx |= px(x + 1, y - 1) << 5;
                cx |= px(x + 0, y - 1) << 6;
                cx |= px(x - 1, y - 1) << 7;
                cx |= px(x - 2, y - 1) << 8;
                cx |= px(x - 3, y - 1) << 9;
            }
            int bit = pixels[(size_t)y * width + x] ? 1 : 0;
            mq_encode_bit(mq, cx_state, cx, bit);
        }
    }
}

static std::vector<uint8_t> mq_encode_generic_template123(int width, int height,
                                                            uint8_t *pixels, int gbtemplate,
                                                            int atx1, int aty1, bool tpgdon)
{
    MQEncoder mq;
    std::vector<uint8_t> cx_state(1u << (gbtemplate == 1 ? 13 : 10), 0);
    mq_encode_generic_template123_into(mq, cx_state, width, height, pixels, gbtemplate,
                                        atx1, aty1, tpgdon);
    mq_flush(mq);
    return mq.out;
}

// Encodes a width x height 1bpp bitmap as arithmetic-coded generic region
// data using GBTEMPLATE 0 with EXTTEMPLATE=1 (the 16-bit, 12-AT-pixel
// extended template, 6.2.5.3's Figure 5). No decoder in this repo's own
// toolchain (jbig2dec, which links pdfium) implements EXTTEMPLATE at all
// -- pdfium's ParseGenericRegion() never reads the flag bit and always
// parses exactly 4 AT pairs for GBTEMPLATE 0, so a real EXTTEMPLATE=1
// segment would desync its parser immediately after the AT-pixel field.
// This layout instead comes from the ITU-T reference encoder/decoder pair
// (JBIG2 Sample Software, Jb2_MQLapper.cpp: CX_Encode's `else` branch,
// used by both MQ_EncImage and MQ_DecImage with the real ExtTemplate flag
// threaded through, not a stub like that same file's TPGDON encoder) --
// the only implementation of this specific spec feature available to port
// from here. Verified by round-trip against a hand-written decoder built
// from mq_decode_bit() (already bit-exact-verified against pdfium's own
// arithmetic coder elsewhere) using this identical context formula, since
// no available real EXTTEMPLATE decoder exists to check against directly.
//
// TPGDON is not supported in combination with this template: Figure 8's
// fixed SLTP pixel pattern would need re-deriving under this template's
// own (different) bit-gathering order, and no available reference gives
// that context number directly.
static std::vector<uint8_t> mq_encode_generic_template0_ext(int width, int height, const uint8_t *pixels,
                                                              const AtPixel at[12])
{
    auto px = [&](int x, int y) -> uint32_t {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return pixels[(size_t)y * width + x];
    };

    MQEncoder mq;
    std::vector<uint8_t> cx_state(1u << 16, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint32_t cx = 0;
            cx |= px(x - 1, y) << 0;
            cx |= px(x + 1, y - 1) << 1;
            cx |= px(x + 0, y - 1) << 2;
            cx |= px(x - 1, y - 1) << 3;
            for (int k = 0; k < 12; k++)
                cx |= px(x + at[k].x, y + at[k].y) << (4 + k);
            int bit = pixels[(size_t)y * width + x] ? 1 : 0;
            mq_encode_bit(mq, cx_state, cx, bit);
        }
    }
    mq_flush(mq);
    return mq.out;
}

// Context bit layouts for the two generic refinement templates (6.3.5.3),
// with GRREFERENCEDX = GRREFERENCEDY = 0 -- always the case for a region
// segment's own refinement (7.4.7/Table 38), as opposed to a text region's
// per-symbol refinement.
//
// These match pdfium's CJBig2_GRRDProc::DecodeTemplate0Unopt /
// DecodeTemplate1Unopt, whose sliding-window `lines[]` accumulators were
// unrolled back into per-pixel coordinates to get them. Template 1 was
// cross-checked against the ITU reference encoder (JBIG2 Sample Software,
// Jb2_MQLapper.cpp: CX_RefEncode case 1) and agrees bit for bit. Its
// case 0 does *not* agree and is not the model here: with rD2_/rD1_/rD0_
// bound to reference rows y-1/y/y+1, that branch reads rD2_ for the
// y+1 terms, so it samples the row above twice and the row below never --
// visible in that rD0_ is passed in and then never referenced there.
//
// Matching a decoder's exact numbering is normally unnecessary: the
// context index is just a name for an adaptive state slot, every slot
// starts at 0, and any injective relabelling keeps encoder and decoder in
// lockstep. TPGRON is what makes it necessary, because it codes its SLTP
// decision under a *fixed* context number (0x0010 here, 0x0008 for
// template 1) that has to collide with the same pixel pattern on both
// sides of the wire.
static const uint32_t GR_TPGRON_CX_T0 = 0x0010;
static const uint32_t GR_TPGRON_CX_T1 = 0x0008;

// Encodes a width x height 1bpp bitmap as arithmetic-coded generic
// refinement region data (6.3), refining it against a same-sized
// reference bitmap. `cur` is modified when tpgron is set: a row coded
// with LTP on leaves every pixel whose reference 3x3 neighbourhood is
// uniform uncoded, and the decoder fills those from the reference, so the
// bitmap the caller ends up holding is the one a decoder reconstructs.
// `shared_cx_state`, when non-null, is reused (and mutated) across
// multiple calls instead of starting fresh each time. Needed when several
// refined instances share one GRCONTEXTS the way a real decoder does --
// see gen_text_region_real()'s per-instance loop, which passes the same
// vector to every refined instance in a region -- and left null (a fresh,
// zeroed context every call) for a single standalone refinement region,
// which owns its GRCONTEXTS outright.
// Core of mq_encode_refinement(), operating on a caller-owned MQEncoder
// instead of a fresh one -- see mq_encode_generic_template0_into()'s
// comment for why (a text region's refined instances, unlike a symbol
// dictionary's Huffman-coded REFAGGNINST==1 case or SBHUFF=1's per-instance
// refinement, share the *same* CJBig2_ArithDecoder as every other
// arithmetic-coded field in the region when SBHUFF=0 -- jbig2_trd_proc.cpp
// DecodeArith's `pGRRD->Decode(pArithDecoder, grContexts)` takes the
// caller's own decoder, not a fresh one restarted at a byte boundary).
static void mq_encode_refinement_into(MQEncoder &mq, std::vector<uint8_t> &cx_state,
                                        int width, int height,
                                        uint8_t *cur, const uint8_t *ref,
                                        int grtemplate, bool tpgron,
                                        int atx1, int aty1, int atx2, int aty2)
{
    auto curpx = [&](int x, int y) -> uint32_t {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return cur[(size_t)y * width + x];
    };
    auto refpx = [&](int x, int y) -> uint32_t {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return ref[(size_t)y * width + x];
    };

    // 6.3.5.6: a pixel is "typical" when the 3x3 reference neighbourhood
    // around it is uniform; on an LTP row those are skipped and take the
    // reference value.
    auto typical = [&](int x, int y) -> bool {
        uint32_t v = refpx(x, y);
        return refpx(x - 1, y - 1) == v && refpx(x, y - 1) == v && refpx(x + 1, y - 1) == v &&
               refpx(x - 1, y)     == v &&                        refpx(x + 1, y)     == v &&
               refpx(x - 1, y + 1) == v && refpx(x, y + 1) == v && refpx(x + 1, y + 1) == v;
    };

    int ltp = 0;
    for (int y = 0; y < height; y++) {
        if (tpgron) {
            // Flip LTP on some rows so both the coded and the predicted
            // path get exercised; SLTP is the *change* in LTP (6.3.5.6).
            int want = (int)(urand() & 1);
            mq_encode_bit(mq, cx_state, grtemplate == 0 ? GR_TPGRON_CX_T0 : GR_TPGRON_CX_T1,
                          want ^ ltp);
            ltp = want;
        }
        for (int x = 0; x < width; x++) {
            if (ltp && typical(x, y)) {
                // Not coded: the decoder assigns the reference pixel, so
                // make `cur` agree before it feeds later contexts.
                cur[(size_t)y * width + x] = (uint8_t)refpx(x, y);
                continue;
            }
            uint32_t cx = 0;
            if (grtemplate == 0) {
                cx |= refpx(x + 1, y + 1) << 0;
                cx |= refpx(x + 0, y + 1) << 1;
                cx |= refpx(x - 1, y + 1) << 2;
                cx |= refpx(x + 1, y + 0) << 3;
                cx |= refpx(x + 0, y + 0) << 4;
                cx |= refpx(x - 1, y + 0) << 5;
                cx |= refpx(x + 1, y - 1) << 6;
                cx |= refpx(x + 0, y - 1) << 7;
                cx |= refpx(x + atx2, y + aty2) << 8;
                cx |= curpx(x - 1, y + 0) << 9;
                cx |= curpx(x + 1, y - 1) << 10;
                cx |= curpx(x + 0, y - 1) << 11;
                cx |= curpx(x + atx1, y + aty1) << 12;
            } else {
                cx |= refpx(x + 1, y + 1) << 0;
                cx |= refpx(x + 0, y + 1) << 1;
                cx |= refpx(x + 1, y + 0) << 2;
                cx |= refpx(x + 0, y + 0) << 3;
                cx |= refpx(x - 1, y + 0) << 4;
                cx |= refpx(x + 0, y - 1) << 5;
                cx |= curpx(x - 1, y + 0) << 6;
                cx |= curpx(x + 1, y - 1) << 7;
                cx |= curpx(x + 0, y - 1) << 8;
                cx |= curpx(x - 1, y - 1) << 9;
            }
            int bit = cur[(size_t)y * width + x] ? 1 : 0;
            mq_encode_bit(mq, cx_state, cx, bit);
        }
    }
}

static std::vector<uint8_t> mq_encode_refinement(int width, int height,
                                                   uint8_t *cur, const uint8_t *ref,
                                                   int grtemplate, bool tpgron,
                                                   int atx1, int aty1, int atx2, int aty2,
                                                   std::vector<uint8_t> *shared_cx_state = nullptr)
{
    MQEncoder mq;
    std::vector<uint8_t> local_cx_state;
    if (!shared_cx_state) {
        local_cx_state.assign(1u << (grtemplate == 0 ? 13 : 10), 0);
        shared_cx_state = &local_cx_state;
    }
    mq_encode_refinement_into(mq, *shared_cx_state, width, height, cur, ref,
                               grtemplate, tpgron, atx1, aty1, atx2, aty2);
    mq_flush(mq);
    return mq.out;
}

// MQ arithmetic *decoder* state, mirroring PDFium's CJBig2_ArithDecoder --
// the byte-consumption twin of MQEncoder above. Used only to measure how
// many bytes of an already-encoded refinement stream a real decode
// consumes (6.4.11.5's RSIZE field needs that exact count; see
// mq_finalize_refinement()), not to recover pixel values for their own
// sake.
struct MQDecoderState {
    const uint8_t *data;
    size_t size;
    size_t byte_idx;
    uint8_t b;
    uint32_t c;
    uint32_t a;
    int ct;
};

static uint8_t mq_dec_cur_byte(const MQDecoderState &d)
{
    return d.byte_idx < d.size ? d.data[d.byte_idx] : 0xFF;
}

static uint8_t mq_dec_next_byte(const MQDecoderState &d)
{
    return d.byte_idx + 1 < d.size ? d.data[d.byte_idx + 1] : 0xFF;
}

// Port of CJBig2_ArithDecoder::BYTEIN.
static void mq_dec_bytein(MQDecoderState &d)
{
    if (d.b == 0xff) {
        uint8_t b1 = mq_dec_next_byte(d);
        if (b1 > 0x8f) {
            d.ct = 8;
        } else {
            d.byte_idx++;
            d.b = b1;
            d.c = d.c + 0xfe00 - ((uint32_t)d.b << 9);
            d.ct = 7;
        }
    } else {
        d.byte_idx++;
        d.b = mq_dec_cur_byte(d);
        d.c = d.c + 0xff00 - ((uint32_t)d.b << 8);
        d.ct = 8;
    }
}

// Port of CJBig2_ArithDecoder's constructor (INITDEC).
static void mq_dec_init(MQDecoderState &d, const uint8_t *data, size_t size)
{
    d.data = data;
    d.size = size;
    d.byte_idx = 0;
    d.b = mq_dec_cur_byte(d);
    d.c = ((uint32_t)(d.b ^ 0xff)) << 16;
    mq_dec_bytein(d);
    d.c = d.c << 7;
    d.ct -= 7;
    d.a = 0x8000;
}

// Port of CJBig2_ArithDecoder::Decode + JBig2ArithCtx::DecodeNMPS/NLPS.
// `cx_state` uses the same packing mq_encode_bit() reads/writes (bits 0-6:
// state index; bit 7: current MPS) -- an internal choice private to this
// encoder/decoder pair, not part of the wire format, so it only needs to
// agree with mq_encode_bit()'s convention, not PDFium's own. In
// particular, the LPS transition is NOT "an NLPS index plus an independent
// switch flag": mq_encode_bit() stores the top byte of MQ_QE_TABLE as
// NLPS-index-with-the-switch-bit-folded-into-bit-7, and applies it via a
// single `(packed >> 24) ^ mps` -- XOR, not OR-after-a-separate-flip. Two
// different fields would only coincidentally agree with that when the
// switch byte happens to be zero, which is common enough among the 47
// table rows to pass casual testing while still being wrong.
static int mq_decode_bit(MQDecoderState &d, std::vector<uint8_t> &cx_state, uint32_t cx)
{
    uint8_t index = cx_state[cx] & 0x7f;
    uint8_t mps = cx_state[cx] & 0x80;
    uint32_t packed = MQ_QE_TABLE[index];
    uint32_t qe = packed & 0xffff;
    uint8_t nmps = (uint8_t)((packed >> 16) & 0xff);
    uint8_t nlps_sw = (uint8_t)((packed >> 24) & 0xff);

    d.a -= qe;
    int bit;
    if ((d.c >> 16) < d.a) {
        if (d.a & 0x8000)
            return mps ? 1 : 0;
        if (d.a < qe) {
            bit = mps ? 0 : 1;
            cx_state[cx] = (uint8_t)(nlps_sw ^ mps);
        } else {
            bit = mps ? 1 : 0;
            cx_state[cx] = (uint8_t)(nmps | mps);
        }
    } else {
        d.c -= d.a << 16;
        if (d.a < qe) {
            bit = mps ? 1 : 0;
            cx_state[cx] = (uint8_t)(nmps | mps);
        } else {
            bit = mps ? 0 : 1;
            cx_state[cx] = (uint8_t)(nlps_sw ^ mps);
        }
        d.a = qe;
    }
    do {
        if (d.ct == 0)
            mq_dec_bytein(d);
        d.a <<= 1;
        d.c <<= 1;
        d.ct--;
    } while (!(d.a & 0x8000));
    return bit;
}

// Decodes a width x height refinement region back from `data`/`size` using
// the exact same per-pixel context formula as mq_encode_refinement(),
// purely to find out how many bytes of `data` a real decode touches (the
// returned image itself is discarded -- see mq_finalize_refinement()).
// `shared_cx_state`: see mq_encode_refinement()'s parameter of the same
// name -- this measurement must share context state across a region's
// refined instances exactly as the real decode being measured does, or
// the two diverge on the second and later instances.
static size_t mq_measure_refinement_bytes(int width, int height, const uint8_t *ref,
                                            int grtemplate, int atx1, int aty1, int atx2, int aty2,
                                            const uint8_t *data, size_t size,
                                            std::vector<uint8_t> *shared_cx_state = nullptr)
{
    auto refpx = [&](int x, int y) -> uint32_t {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return ref[(size_t)y * width + x];
    };
    std::vector<uint8_t> cur((size_t)width * height, 0);
    auto curpx = [&](int x, int y) -> uint32_t {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return cur[(size_t)y * width + x];
    };

    MQDecoderState dec;
    mq_dec_init(dec, data, size);
    std::vector<uint8_t> local_cx_state;
    if (!shared_cx_state) {
        local_cx_state.assign(1u << (grtemplate == 0 ? 13 : 10), 0);
        shared_cx_state = &local_cx_state;
    }
    std::vector<uint8_t> &cx_state = *shared_cx_state;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint32_t cx = 0;
            if (grtemplate == 0) {
                cx |= refpx(x + 1, y + 1) << 0;
                cx |= refpx(x + 0, y + 1) << 1;
                cx |= refpx(x - 1, y + 1) << 2;
                cx |= refpx(x + 1, y + 0) << 3;
                cx |= refpx(x + 0, y + 0) << 4;
                cx |= refpx(x - 1, y + 0) << 5;
                cx |= refpx(x + 1, y - 1) << 6;
                cx |= refpx(x + 0, y - 1) << 7;
                cx |= refpx(x + atx2, y + aty2) << 8;
                cx |= curpx(x - 1, y + 0) << 9;
                cx |= curpx(x + 1, y - 1) << 10;
                cx |= curpx(x + 0, y - 1) << 11;
                cx |= curpx(x + atx1, y + aty1) << 12;
            } else {
                cx |= refpx(x + 1, y + 1) << 0;
                cx |= refpx(x + 0, y + 1) << 1;
                cx |= refpx(x + 1, y + 0) << 2;
                cx |= refpx(x + 0, y + 0) << 3;
                cx |= refpx(x - 1, y + 0) << 4;
                cx |= refpx(x + 0, y - 1) << 5;
                cx |= curpx(x - 1, y + 0) << 6;
                cx |= curpx(x + 1, y - 1) << 7;
                cx |= curpx(x + 0, y - 1) << 8;
                cx |= curpx(x - 1, y - 1) << 9;
            }
            cur[(size_t)y * width + x] = (uint8_t)mq_decode_bit(dec, cx_state, cx);
        }
    }
    return dec.byte_idx;
}

// Trims (or extends) mq_encode_refinement()'s raw output to exactly the
// byte count a real decoder will consume, and reports that count via
// `out_rsize` (already +2, matching the fixed post-decode skip every
// caller of this -- see gen_text_region_real() -- applies). Appending two
// 0xFF bytes before measuring makes the count provably finite: BYTEIN
// permanently freezes its byte position the first time it sees 0xFF
// followed by a byte > 0x8F (Annex E.2.9's termination convention), which
// those two bytes always satisfy, so measurement never reads past them.
// `shared_cx_state`: see mq_encode_refinement()'s parameter of the same
// name, forwarded to mq_measure_refinement_bytes() unchanged.
static std::vector<uint8_t> mq_finalize_refinement(int width, int height, const uint8_t *ref,
                                                      int grtemplate, int atx1, int aty1,
                                                      int atx2, int aty2,
                                                      const std::vector<uint8_t> &coded,
                                                      uint32_t *out_rsize,
                                                      std::vector<uint8_t> *shared_cx_state = nullptr)
{
    std::vector<uint8_t> measure_buf = coded;
    measure_buf.push_back(0xFF);
    measure_buf.push_back(0xFF);

    size_t consumed = mq_measure_refinement_bytes(width, height, ref, grtemplate,
                                                    atx1, aty1, atx2, aty2,
                                                    measure_buf.data(), measure_buf.size(),
                                                    shared_cx_state);
    std::vector<uint8_t> out(measure_buf.begin(), measure_buf.begin() + (ptrdiff_t)consumed);
    *out_rsize = (uint32_t)(consumed + 2);
    return out;
}

// Same idea as mq_measure_refinement_bytes()/mq_finalize_refinement(), for
// a plain generic-region bitmap instead of a refinement -- used by
// gen_segment_generic_region()'s unknown-length case (7.2.7/7.4.6.4),
// where a real decoder locates the segment's true end by decoding exactly
// GBW x GBH pixels and then reading whatever two bytes follow as the
// terminator: mq_flush()'s own trailing bytes are ordinary MQ-coder flush
// overhead, not engineered to land the marker at any particular position,
// so the number of "real" bytes a decode actually touches is smaller than
// (and not reliably a fixed offset from) mq.out.size() -- it has to be
// measured the same way real content does, not assumed.
static size_t mq_measure_generic_bytes(int width, int height, int gbtemplate,
                                         int atx1, int aty1, int atx2, int aty2,
                                         int atx3, int aty3, int atx4, int aty4,
                                         bool tpgdon, const uint8_t *data, size_t size)
{
    std::vector<uint8_t> pixels((size_t)width * height, 0);
    auto px = [&](int x, int y) -> uint32_t {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return pixels[(size_t)y * width + x];
    };

    MQDecoderState dec;
    mq_dec_init(dec, data, size);
    std::vector<uint8_t> cx_state(1u << (gbtemplate == 0 ? 16 : gbtemplate == 1 ? 13 : 10), 0);

    int ltp = 0;
    for (int y = 0; y < height; y++) {
        if (tpgdon) {
            int bit = mq_decode_bit(dec, cx_state, GD_TPGDON_CX[gbtemplate]);
            ltp ^= bit;
        }
        if (ltp) {
            for (int x = 0; x < width; x++)
                pixels[(size_t)y * width + x] = (uint8_t)px(x, y - 1);
            continue;
        }
        for (int x = 0; x < width; x++) {
            uint32_t cx = 0;
            if (gbtemplate == 0) {
                cx |= px(x - 1, y) << 0;
                cx |= px(x - 2, y) << 1;
                cx |= px(x - 3, y) << 2;
                cx |= px(x - 4, y) << 3;
                cx |= px(x + atx1, y + aty1) << 4;
                cx |= px(x + 2, y - 1) << 5;
                cx |= px(x + 1, y - 1) << 6;
                cx |= px(x + 0, y - 1) << 7;
                cx |= px(x - 1, y - 1) << 8;
                cx |= px(x - 2, y - 1) << 9;
                cx |= px(x + atx2, y + aty2) << 10;
                cx |= px(x + atx3, y + aty3) << 11;
                cx |= px(x + 1, y - 2) << 12;
                cx |= px(x + 0, y - 2) << 13;
                cx |= px(x - 1, y - 2) << 14;
                cx |= px(x + atx4, y + aty4) << 15;
            } else if (gbtemplate == 1) {
                cx |= px(x - 1, y) << 0;
                cx |= px(x - 2, y) << 1;
                cx |= px(x - 3, y) << 2;
                cx |= px(x + atx1, y + aty1) << 3;
                cx |= px(x + 2, y - 1) << 4;
                cx |= px(x + 1, y - 1) << 5;
                cx |= px(x + 0, y - 1) << 6;
                cx |= px(x - 1, y - 1) << 7;
                cx |= px(x - 2, y - 1) << 8;
                cx |= px(x + 2, y - 2) << 9;
                cx |= px(x + 1, y - 2) << 10;
                cx |= px(x + 0, y - 2) << 11;
                cx |= px(x - 1, y - 2) << 12;
            } else if (gbtemplate == 2) {
                cx |= px(x - 1, y) << 0;
                cx |= px(x - 2, y) << 1;
                cx |= px(x + atx1, y + aty1) << 2;
                cx |= px(x + 1, y - 1) << 3;
                cx |= px(x + 0, y - 1) << 4;
                cx |= px(x - 1, y - 1) << 5;
                cx |= px(x - 2, y - 1) << 6;
                cx |= px(x + 1, y - 2) << 7;
                cx |= px(x + 0, y - 2) << 8;
                cx |= px(x - 1, y - 2) << 9;
            } else {
                cx |= px(x - 1, y) << 0;
                cx |= px(x - 2, y) << 1;
                cx |= px(x - 3, y) << 2;
                cx |= px(x - 4, y) << 3;
                cx |= px(x + atx1, y + aty1) << 4;
                cx |= px(x + 1, y - 1) << 5;
                cx |= px(x + 0, y - 1) << 6;
                cx |= px(x - 1, y - 1) << 7;
                cx |= px(x - 2, y - 1) << 8;
                cx |= px(x - 3, y - 1) << 9;
            }
            pixels[(size_t)y * width + x] = (uint8_t)mq_decode_bit(dec, cx_state, cx);
        }
    }
    return dec.byte_idx;
}

// Trims mq_encode_generic_template0()/mq_encode_generic_template123()'s raw
// output to exactly the byte count a real decode of GBW x GBH pixels
// touches -- see mq_measure_generic_bytes()'s comment for why that's
// smaller than the raw output. Used only by the unknown-length generic
// region case, which needs its own terminator placed at that exact byte,
// not the flush's actual end.
static std::vector<uint8_t> mq_finalize_generic(int width, int height, int gbtemplate,
                                                   int atx1, int aty1, int atx2, int aty2,
                                                   int atx3, int aty3, int atx4, int aty4,
                                                   bool tpgdon, const std::vector<uint8_t> &coded)
{
    std::vector<uint8_t> measure_buf = coded;
    measure_buf.push_back(0xFF);
    measure_buf.push_back(0xFF);
    size_t consumed = mq_measure_generic_bytes(width, height, gbtemplate,
                                                 atx1, aty1, atx2, aty2, atx3, aty3, atx4, aty4,
                                                 tpgdon, measure_buf.data(), measure_buf.size());
    return std::vector<uint8_t>(measure_buf.begin(), measure_buf.begin() + (ptrdiff_t)consumed);
}

void fill_random_pattern(uint8_t *buf, size_t len);

// Appends a block of random bytes representing entropy-coded content
// (arithmetic- or Huffman-coded bitmaps, tables, etc.) that this generator
// does not attempt to synthesize faithfully.
static void append_random_payload(std::vector<uint8_t> &d, uint32_t max_len)
{
    uint32_t len = urand() % (max_len + 1);
    size_t off = d.size();
    d.resize(off + len);
    fill_random_pattern(d.data() + off, len);
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

// 8.2 step 5a/5c/5d: the region segment types combined straight into the
// page buffer with their own external combination operator. The
// *intermediate* types (4, 20, 36, 40) are excluded deliberately: those
// are decoded into an auxiliary buffer and only reach the page later, via
// whatever refinement region refers to them (step 5b/5e).
static bool is_immediate_direct_region(uint8_t type)
{
    switch (type) {
    case SEG_IMMEDIATE_TEXT:
    case SEG_IMMEDIATE_LOSSLESS_TEXT:
    case SEG_IMMEDIATE_HALFTONE:
    case SEG_IMMEDIATE_LOSSLESS_HALFTONE:
    case SEG_IMMEDIATE_GENERIC:
    case SEG_IMMEDIATE_LOSSLESS_GENERIC:
    case SEG_IMMEDIATE_GENERIC_REFINEMENT:
    case SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT:
        return true;
    default:
        return false;
    }
}

// One symbol dictionary entry's exact pixels, in export order -- see the
// `symbols` field below.
struct ExportedSymbol {
    uint32_t w, h;
    std::vector<uint8_t> px;   // row-major, one byte per pixel, 0 or 1
};

// A segment already emitted earlier in this file, available for later
// segments to refer to (7.2.5: a segment may only refer to lower-numbered
// segments, which increasing g_next_segment_number assignment guarantees).
struct GeneratedSegment {
    uint32_t number;
    uint8_t type;
    uint32_t page;
    bool colored;         // true if a region segment set COLEXTFLAG (7.4.1.5)
    int combop;            // region external combination operator (0-4), or -1 if not a region segment
    uint32_t nrefs;        // size of this segment's own referred-to list
    uint32_t num_symbols;  // symbol dictionary: SDNUMEXSYMS, else 0
    bool ext_template;     // true if a generic region set EXTTEMPLATE (7.4.6.2)
    // The bitmap this segment decodes to, when the generator coded real
    // content and so knows it exactly (empty when the data part was random
    // payload). One byte per pixel, row-major, bw x bh. Only an
    // *intermediate* region keeps its bitmap as a segment result later
    // segments can reach: an immediate region is composed onto the page and
    // its buffer released (7.4.6.4/7.4.7.6), so only the intermediate types
    // are worth drawing from -- see gen_segment_refinement_region(), which
    // uses one as a real GRREFERENCE.
    uint32_t bw;
    uint32_t bh;
    std::vector<uint8_t> bitmap;
    // A real symbol dictionary's exported glyphs, in the same order a
    // decoder assigns them to SBSYMS (7.4.3.1.6) -- empty for anything
    // else, or a structural (random-payload) symbol dictionary. Lets a
    // text region that refers to this dictionary refine an existing glyph
    // against its real pixels rather than an invented one -- see
    // gen_text_region_real()'s SBREFINE=1 path.
    std::vector<ExportedSymbol> symbols;
    // A real tables (Annex B.2) segment's one directly-encodable line --
    // the last of its `nlines` ordinary lines, val = HTLOW + nlines - 1,
    // encodable with a zero range-bits offset -- so a later real-content
    // generator can reference this segment as a genuine "user-supplied"
    // (selector 3) Huffman table instead of only a standard one. Empty for
    // anything but a SEG_TABLES segment. See gen_segment_tables()'s comment.
    std::vector<StdHuffLine> table_rows;
    // A real pattern dictionary's own HDPW (0 for anything but a
    // SEG_PATTERN_DICTIONARY segment with real content). Pattern i is the
    // hdpw x bh slice of `bitmap` at column i*hdpw (bw/hdpw of them in
    // total) -- see gen_segment_pattern_dict()'s comment on `bitmap`. Lets
    // a halftone region recover individual patterns from the collective
    // bitmap its referred-to pattern dictionary already decodes to.
    uint32_t hdpw = 0;
};

Knubs knubs(void)
{
    Knubs k;
    k.page_number_known = urand() % 2;
    k.use_12_AT = urand() % 2;
    k.colored_region = urand() % 2;
    return k;
}

// This run's page geometry, and a running model of the page buffer's
// contents. Every other page field gen_segment_page_info() derives after
// the fact (main() builds that segment's bytes last, once content is
// known), but the geometry has to be settled *before* content generation:
// a region's placement is only meaningful relative to a page that already
// has a size, and one segment type -- a generic refinement region that
// refers to no other segment (7.4.7.4) -- reads the page buffer back as
// its GRREFERENCE, so it can only encode real content if it knows both
// where the page is and what is already on it.
uint32_t g_page_width = 0;
uint32_t g_page_height = 0;
bool g_page_default_pixel = false;
// One byte per pixel, row-major, g_page_width x g_page_height: exactly
// what a decoder's page buffer holds after every content segment emitted
// so far has been composed onto it (8.2 step 5). pdfium's page buffer is
// supplied by the harness (CJBig2_Context::GetFirstPage sets
// buf_specified_), so the "page grows on an end-of-stripe" path
// (page_->Expand) is guarded off and the page keeps these exact
// dimensions for the whole file -- nothing here has to model striping.
std::vector<uint8_t> g_page_bitmap;
// Cleared once anything is emitted whose effect on the page buffer this
// generator cannot reproduce -- a region segment carrying random payload,
// say. Such a segment also makes the whole file fail to decode (pdfium
// stops at the first failing segment), so the page state after it is moot;
// this only stops a later segment from *claiming* to know it.
bool g_page_state_known = true;

// 7.4.8.1/.2 + case 48's page_->Fill(default pixel): the state a decoder's
// page buffer is in before any region segment composes onto it.
static void page_state_init(uint32_t width, uint32_t height, bool default_pixel)
{
    g_page_width = width;
    g_page_height = height;
    g_page_default_pixel = default_pixel;
    g_page_bitmap.assign((size_t)width * height, default_pixel ? 1 : 0);
    g_page_state_known = true;
}

// Draws 7.4.8.1/.2's page width and height, and 7.4.8.5 bit 2's default
// pixel value. Unlike a region's declared size, these drive an actual
// page-buffer allocation, so the *product* matters: two independent
// 1..0x10000 draws reach 2^32 pixels and decoders refuse the page outright
// (pdfium caps a bitmap at INT_MAX - 31), killing the file before any
// content segment is reached. Cap the area instead, letting either
// dimension still span its full range as long as the other gives way --
// and keep the total small enough that g_page_bitmap's byte-per-pixel
// model stays cheap.
static void choose_page_geometry(void)
{
    static const uint32_t MAX_PAGE_PIXELS = 1u << 22;   // 4M pixels
    uint32_t width = 1 + (urand() % 0x10000);
    uint32_t max_height = MAX_PAGE_PIXELS / width;
    if (max_height == 0)
        max_height = 1;
    if (max_height > 0x10000)
        max_height = 0x10000;
    uint32_t height = 1 + (urand() % max_height);
    page_state_init(width, height, (urand() & 1) != 0);
}

std::vector<uint8_t> stream;

size_t g_segment_len = 0;

// Every segment emitted so far this run, in generation (= segment-number)
// order, so later segments can refer to real earlier ones instead of
// picking meaningless numbers.
std::vector<GeneratedSegment> g_prior_segments;
uint32_t g_next_segment_number = 0;

// Returns the segment numbers of all prior segments of a given type, in
// generation order.
static std::vector<uint32_t> segment_numbers_of_type(const std::vector<GeneratedSegment> &prior,
                                                       uint8_t type)
{
    std::vector<uint32_t> out;
    for (const auto &seg : prior)
        if (seg.type == type)
            out.push_back(seg.number);
    return out;
}

// Reference-selection strategies. Drawing a referred-to segment uniformly
// at random spreads the choice over the middle of the candidate pool and
// reaches its endpoints only by luck: for a pool of n the oldest and the
// newest each come up 1/n of the time, and the largest legal reference
// count almost never. Those extremes are the interesting ones. The newest
// candidate is the segment most likely to still be live in whatever state
// a decoder carries forward; the oldest is the most likely to have been
// retired or evicted; an empty list exercises the "no referred-to segment"
// fallback that several segment types define separately (7.4.7.4's page-
// buffer default, say); and a maximal list stresses the referred-to
// bookkeeping and the long-form count encoding in 7.2.4. Picking a named
// strategy first and then filling it gives each shape a fixed share of
// runs rather than a vanishing one.
enum RefStrategy {
    REF_NONE,        // empty list, where the segment type permits it
    REF_OLDEST,      // lowest-numbered candidate
    REF_NEWEST,      // highest-numbered candidate
    REF_RANDOM_ONE,  // any single candidate
    REF_MAX          // as many as the segment type permits
};

// `pool` is in generation order, so front() is the oldest candidate and
// back() the newest, and every entry is lower-numbered than the segment
// being built (7.2.5). min_refs is the fewest the segment type permits --
// 0 where a reference is optional, 1 where one is required -- and
// max_refs the most it can use. An empty pool yields an empty list even
// when min_refs is 1: callers that require a reference have no candidate
// to offer and handle that case themselves. The result is ascending,
// matching the order the numbers are written in.
static std::vector<uint32_t> pick_refs(const std::vector<uint32_t> &pool,
                                        size_t min_refs, size_t max_refs)
{
    if (pool.empty() || max_refs == 0)
        return std::vector<uint32_t>();
    if (max_refs > pool.size())
        max_refs = pool.size();

    RefStrategy choices[5];
    size_t n = 0;
    if (min_refs == 0)
        choices[n++] = REF_NONE;
    if (min_refs <= 1) {
        choices[n++] = REF_OLDEST;
        choices[n++] = REF_NEWEST;
        choices[n++] = REF_RANDOM_ONE;
    }
    if (max_refs > 1)
        choices[n++] = REF_MAX;

    switch (n ? choices[urand() % n] : REF_MAX) {
    case REF_NONE:
        return std::vector<uint32_t>();
    case REF_OLDEST:
        return std::vector<uint32_t>(1, pool.front());
    case REF_NEWEST:
        return std::vector<uint32_t>(1, pool.back());
    case REF_RANDOM_ONE:
        return std::vector<uint32_t>(1, pool[urand() % pool.size()]);
    case REF_MAX:
    default:
        // The newest max_refs candidates, ascending; the single-reference
        // strategies above already cover the oldest end of the pool.
        return std::vector<uint32_t>(pool.end() - (ptrdiff_t)max_refs, pool.end());
    }
}

// Two pointer streams filled from gensegment()'s output: one for the
// header parts, one for the data parts, in generation order. gensegment()
// assigns strictly increasing segment numbers in this same order, which is
// what Annex D.1/D.2 require.
std::vector<std::vector<uint8_t> *> header_streams;
std::vector<std::vector<uint8_t> *> data_streams;

// The organization chosen for this run. gensegment() consults it because
// D.2's random-access layout constrains what a segment header may declare
// (see the unknown-length decision there).
Organization g_organisation = ORG_SEQUENTIAL;

Organization choose_organisation(void)
{
    return (Organization)(urand() % 3);
}

// number_of_pages: how many pages the caller actually generated. D.4.3's
// count is a claim about the file's contents, so it comes from the caller
// rather than being drawn at random -- the same reason the 12-AT and
// coloured-region flags below are derived from real content.
//
// write_header is independent of org: normally (D.4) a file header is
// present for sequential/random-access and absent for embedded, but
// main()'s --no-header can suppress it regardless of which organization
// was chosen -- some setups want to feed a decoder a headerless stream
// laid out in an otherwise-arbitrary organization (e.g. random-access's
// headers-then-data shape), not just the one organization the spec
// actually pairs with "no header".
void genheader(Organization org, Knubs k, uint32_t number_of_pages, bool write_header)
{
    if (!write_header) {
        printf("no file header written (org=%d)\n", (int)org);
        return;
    }

    // D.4/D.4.1: the header's first field is an 8-byte ID string, without
    // which nothing downstream can recognise the file as JBIG2 or find the
    // first segment header.
    static const uint8_t id_string[8] = { 0x97, 0x4A, 0x42, 0x32, 0x0D, 0x0A, 0x1A, 0x0A };
    append(stream, id_string, sizeof(id_string));

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
// segment_number and refs come from the caller rather than being invented
// here: segment_number is assigned by gensegment() from a monotonically
// increasing counter (D.1/D.2), and refs are real, earlier segment numbers
// chosen by the data handler to match what its own fields actually need
// (7.2.5 requires referring only to lower-numbered segments, which holding
// to increasing segment numbers guarantees).
std::vector<uint8_t> gensegmentheader(uint8_t segment_type, uint32_t segment_number,
                                       const std::vector<uint32_t> &refs, uint32_t data_len,
                                       size_t *out_len, int32_t forced_page = -1,
                                       bool unknown_len = false)
{
    std::vector<uint8_t> hdr;

    uint32_t R = (uint32_t)refs.size();
    bool big_page = urand() & 1;            // page association field size

    // 7.2.3: segment header flags.
    uint8_t flags = segment_type & 0x3F;    // bits 0-5: segment type
    if (big_page)
        flags |= 0x40;                      // bit 6: 4-byte page association
    if (urand() & 1)
        flags |= 0x80;                      // bit 7: deferred non-retain

    // 7.2.7: only an immediate generic region may declare an unknown data
    // length, and only when the caller actually appended the trailer that
    // makes the true length recoverable -- hence the decision arriving as
    // a parameter rather than being taken here.
    if (unknown_len && segment_type != SEG_IMMEDIATE_GENERIC) {
        fprintf(stderr, "gensegmentheader: unknown length illegal for segment type %u\n", segment_type);
        exit(1);
    }
    uint32_t seg_len = unknown_len ? 0xFFFFFFFFu : data_len;

    printf("segment number = %u, type = %u, R = %u\n", segment_number, segment_type, R);
    printf("refs =");
    for (uint32_t ref : refs)
        printf(" %u", ref);
    printf("\n");

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

    // 7.2.5: referred-to segment numbers; field size depends on this
    // segment's own number.
    for (uint32_t ref : refs) {
        if (segment_number <= 256)
            hdr.push_back((uint8_t)ref);
        else if (segment_number <= 65536)
            put_be16(hdr, (uint16_t)ref);
        else
            put_be32(hdr, ref);
    }

    // 7.2.6: page association. 0 = not associated with any page; 1 is the first page.
    uint32_t page;
    if (forced_page >= 0) {
        page = (uint32_t)forced_page;
    } else if (urand() & 1) {
        page = 0;
    } else {
        // The 1-byte form must stay in 1..255: 1 + (urand() & 0xFF) reaches
        // 256, which push_back() below truncates to 0 -- "not associated
        // with any page" (7.2.6), silently contradicting the printf.
        page = big_page ? 1 + (urand() & 0x0FFFFFFFu) : 1 + (urand() % 255);
    }
    if (big_page)
        put_be32(hdr, page);
    else
        hdr.push_back((uint8_t)page);
    printf("page association = %u\n", page);

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

// A segment-data handler builds the data part for one segment type, plus
// the list of earlier segment numbers it actually refers to (if any),
// chosen from `prior` to match what its own fields need (e.g. a text
// region's referred-to symbol dictionaries, or a Huffman selector's
// user-supplied table). Table is indexed by SegmentType; entries that are
// nullptr are "unimplemented" and gensegmentdata() falls back to random
// bytes with no refs.
struct SegResult {
    std::vector<uint8_t> data;
    std::vector<uint32_t> refs;
    bool colored = false;       // true if this is a region segment with COLEXTFLAG set
    int combop = -1;            // region external combination operator (0-4), or -1 if not a region segment
    uint32_t num_symbols = 0;   // symbol dictionary: SDNUMEXSYMS, else 0
    bool ext_template = false;  // true if this is a generic region with EXTTEMPLATE set
    // Generic regions only, for the unknown-length trailer gensegment()
    // appends (7.2.7). Reported by the handler rather than recovered by
    // re-parsing the bytes it just wrote, so the trailer cannot drift out
    // of step with the data part's actual layout or contents.
    bool mmr = false;           // MMR (1) vs template-based arithmetic (0) coding
    uint32_t region_rows = 0;   // rows of bitmap data actually encoded
    // The exact bitmap this segment's data part decodes to, when real
    // content was coded; empty otherwise. Carried up to GeneratedSegment so
    // a later refinement region can use it as GRREFERENCE.
    uint32_t bw = 0;
    uint32_t bh = 0;
    std::vector<uint8_t> bitmap = {};
    // Where a region segment declared itself on the page (7.4.1.3/.1.4),
    // so gensegment() can compose `bitmap` onto its model of the page
    // buffer exactly where a decoder will. Meaningless when combop is -1
    // (not a region segment).
    int32_t region_x = 0;
    int32_t region_y = 0;
    // A real symbol dictionary's exported glyphs (see GeneratedSegment's
    // field of the same name); empty for everything else.
    std::vector<ExportedSymbol> symbols = {};
    // A real tables segment's directly-encodable line (see GeneratedSegment's
    // field of the same name); empty for everything else.
    std::vector<StdHuffLine> table_rows = {};
    // A real pattern dictionary's HDPW (see GeneratedSegment's field of the
    // same name); 0 for everything else.
    uint32_t hdpw = 0;
};
typedef SegResult (*SegHandler)(const std::vector<GeneratedSegment> &prior);

struct RegionInfo {
    std::vector<uint8_t> bytes;
    bool colored;
    uint8_t combop;
    uint32_t width;
    uint32_t height;
    // 7.4.1.3/.1.4, as the decoder reads them: a raw 32-bit field stored
    // into an int32_t, so a large unsigned value here really is a negative
    // placement. Carried out so gensegment() can compose this region onto
    // its model of the page buffer at the same spot a decoder will.
    int32_t x;
    int32_t y;
};

SegResult gen_segment_page_info(const std::vector<GeneratedSegment> &prior);
SegResult gen_segment_extension(const std::vector<GeneratedSegment> &prior);
SegResult gen_segment_pattern_dict(const std::vector<GeneratedSegment> &prior);
RegionInfo gen_segment_region_info(bool force_replace = false, uint32_t max_dim = 0x10000,
                                     uint32_t force_w = 0, uint32_t force_h = 0,
                                     int64_t force_x = INT64_MIN, int64_t force_y = INT64_MIN);
SegResult gen_segment_symbol_dict(const std::vector<GeneratedSegment> &prior);
SegResult gen_symbol_dict_real(void);
SegResult gen_segment_text_region(const std::vector<GeneratedSegment> &prior);
SegResult gen_text_region_real(const std::vector<GeneratedSegment> &prior, bool sbrefine);
SegResult gen_text_region_real_arith(const std::vector<GeneratedSegment> &prior, bool sbrefine);
SegResult gen_segment_halftone_region(const std::vector<GeneratedSegment> &prior);
SegResult gen_segment_generic_region(const std::vector<GeneratedSegment> &prior);
SegResult gen_segment_refinement_region(const std::vector<GeneratedSegment> &prior);
SegResult gen_segment_profiles(const std::vector<GeneratedSegment> &prior);
SegResult gen_segment_tables(const std::vector<GeneratedSegment> &prior);
SegResult gen_segment_colour_palette(const std::vector<GeneratedSegment> &prior);
SegResult gen_segment_end_of_stripe(const std::vector<GeneratedSegment> &prior);

static const size_t SEG_HANDLER_COUNT = 63;
static SegHandler seg_handlers[SEG_HANDLER_COUNT] = { nullptr };

void init_seg_handlers(void)
{
    seg_handlers[SEG_SYMBOL_DICTIONARY] = gen_segment_symbol_dict;
    seg_handlers[SEG_PAGE_INFORMATION] = gen_segment_page_info;
    seg_handlers[SEG_EXTENSION] = gen_segment_extension;
    seg_handlers[SEG_PATTERN_DICTIONARY] = gen_segment_pattern_dict;
    seg_handlers[SEG_PROFILES] = gen_segment_profiles;
    seg_handlers[SEG_TABLES] = gen_segment_tables;
    seg_handlers[SEG_COLOUR_PALETTE] = gen_segment_colour_palette;
    seg_handlers[SEG_END_OF_STRIPE] = gen_segment_end_of_stripe;

    seg_handlers[SEG_INTERMEDIATE_TEXT] = gen_segment_text_region;
    seg_handlers[SEG_IMMEDIATE_TEXT] = gen_segment_text_region;
    seg_handlers[SEG_IMMEDIATE_LOSSLESS_TEXT] = gen_segment_text_region;
    seg_handlers[SEG_INTERMEDIATE_HALFTONE] = gen_segment_halftone_region;
    seg_handlers[SEG_IMMEDIATE_HALFTONE] = gen_segment_halftone_region;
    seg_handlers[SEG_IMMEDIATE_LOSSLESS_HALFTONE] = gen_segment_halftone_region;
    seg_handlers[SEG_INTERMEDIATE_GENERIC] = gen_segment_generic_region;
    seg_handlers[SEG_IMMEDIATE_GENERIC] = gen_segment_generic_region;
    seg_handlers[SEG_IMMEDIATE_LOSSLESS_GENERIC] = gen_segment_generic_region;
    seg_handlers[SEG_INTERMEDIATE_GENERIC_REFINEMENT] = gen_segment_refinement_region;
    seg_handlers[SEG_IMMEDIATE_GENERIC_REFINEMENT] = gen_segment_refinement_region;
    seg_handlers[SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT] = gen_segment_refinement_region;
}

// 7.4.1: region segment information field (Figure 30). This common
// 17-byte prefix opens the data part of every region segment (text,
// halftone, generic, generic refinement). Field order:
//   width, height, X location, Y location, flags.
// force_replace: 7.4.7.5 step 1 requires a refinement region that refers to
// no other region segment to use REPLACE as its external combination
// operator; callers for which that applies pass true.
// max_dim: caps width/height. Callers that go on to synthesize and encode
// real content (e.g. real MMR) need the declared size to match what they
// can actually afford to generate/encode; the 0x10000 default matches the
// page-size cap used elsewhere for callers that don't have that constraint.
RegionInfo gen_segment_region_info(bool force_replace, uint32_t max_dim,
                                     uint32_t force_w, uint32_t force_h,
                                     int64_t force_x, int64_t force_y)
{
    std::vector<uint8_t> d;
    // A caller passes force_w/force_h when the region's size is not free to
    // choose: a refinement region reusing another segment's bitmap as
    // GRREFERENCE must declare that bitmap's exact size, since the decoder
    // samples the reference over this region's own extent (6.3.5.3).
    //
    // A freely-chosen size is capped by *area*, not just per dimension, for
    // the same reason choose_page_geometry() caps the page: a decoder has
    // to walk every pixel of whatever this declares. pdfium's own guard
    // (CJBig2_Image::IsValidImageSize) only bounds each dimension at 65535
    // and never the product, so two independent draws land squarely in a
    // band -- 20480 x 4095, say -- that is accepted and then costs ~83
    // million arithmetic decisions, seconds per segment, for a region whose
    // content is random payload nobody can decode anyway. Either dimension
    // may still span the full range as long as the other gives way, so
    // wide-and-short and tall-and-narrow shapes stay reachable.
    static const uint32_t MAX_REGION_PIXELS = 1u << 18;   // 256K pixels
    uint32_t width, height;
    if (max_dim >= 0x10000 && (urand() % 16) == 0) {
        // Keep the "declared size a decoder must refuse outright" case:
        // past 65535 IsValidImageSize rejects before anything is allocated
        // or decoded, so this stays cheap -- unlike the accepted-but-huge
        // band above, which is what the area cap exists to avoid.
        width = force_w ? force_w : 0x10000 + (urand() % 1000);
        height = force_h ? force_h : 0x10000 + (urand() % 1000);
    } else if (urand() & 1) {
        // Width first, height yielding: wide-and-short.
        width = force_w ? force_w : 1 + (urand() % max_dim);
        if (force_h) {
            height = force_h;
        } else {
            uint32_t max_h = MAX_REGION_PIXELS / width;
            if (max_h == 0)
                max_h = 1;
            if (max_h > max_dim)
                max_h = max_dim;
            height = 1 + (urand() % max_h);
        }
    } else {
        // Height first, width yielding: tall-and-narrow. Drawing the same
        // dimension first every time would make the *other* one collapse to
        // a handful of pixels whenever the first came out large, so one
        // orientation would effectively never appear.
        height = force_h ? force_h : 1 + (urand() % max_dim);
        if (force_w) {
            width = force_w;
        } else {
            uint32_t max_w = MAX_REGION_PIXELS / height;
            if (max_w == 0)
                max_w = 1;
            if (max_w > max_dim)
                max_w = max_dim;
            width = 1 + (urand() % max_w);
        }
    }
    put_be32(d, width);                       // 7.4.1.1: bitmap width
    put_be32(d, height);                      // 7.4.1.2: bitmap height
    // 7.4.1.3: X location. ParseRegionInfo (jbig2_context.cpp) reads this
    // as a raw 32-bit value stored straight into an int32_t field, so a
    // large unsigned value here decodes as negative, clipping this
    // region's left edge off the page when it's composed. A fully-on-page
    // region always has its source clip-start byte-aligned to bit 0
    // (CJBig2_Image::ComposeToInternal's `xs0` stays 0), so without this
    // the "source clip start isn't word-aligned and starts further into
    // its word than the destination" cases (s1 > d1, jbig2_image.cpp's
    // kDest{Aligned,NotAligned}Src...GreaterThanDest) can never run --
    // SubImage() (used by a reference-less refinement region to sample the
    // page buffer) already no-ops safely on a negative x, so this is free.
    // Whether that lands in the single-word or multi-word case (of either
    // family) depends on how much survives the clip, not the clip amount
    // itself: only when the *post-clip* width still exceeds 32 pixels does
    // the multi-word ("NotAligned") family ever run, so a wide-enough
    // region caps its own clip small enough to guarantee that instead of
    // leaving it to chance -- a clip comparable to the whole width, or a
    // region under 64 pixels wide to begin with, can only ever land back
    // in the single-word family regardless of the clip's own alignment.
    // force_x/force_y pin the placement instead, for a caller that needs
    // the region to land somewhere specific on the page -- a
    // reference-less refinement region taking GRREFERENCE from the page
    // buffer has to sit fully inside it, or the reference it encoded
    // against is not the sub-image the decoder will sample.
    int32_t x;
    if (force_x != INT64_MIN) {
        x = (int32_t)force_x;
    } else if ((urand() & 1) && width >= 64) {
        x = -(int32_t)(1 + urand() % 31);       // remaining width in [33,63]
    } else if (urand() & 1) {
        x = -(int32_t)(1 + urand() % 32);       // remaining width often <=32
    } else {
        x = (int32_t)(urand() & 0xFFFF);
    }
    int32_t y = force_y != INT64_MIN ? (int32_t)force_y : (int32_t)(urand() & 0xFFFF);
    put_be32(d, (uint32_t)x);                 // 7.4.1.3: X location
    put_be32(d, (uint32_t)y);                 // 7.4.1.4: Y location
    // 7.4.1.5 flags: bits 0-2 external combination operator (0 OR,
    // 1 AND, 2 XOR, 3 XNOR, 4 REPLACE); bit 3 COLEXTFLAG; bits 4-7
    // reserved, must be 0.
    bool color = (urand() & 1) != 0;
    uint8_t combop;
    if (force_replace)
        combop = 4;                                   // REPLACE (7.4.7.5 step 1)
    else
        combop = color ? 4 : (uint8_t)(urand() % 5);   // COLEXTFLAG -> REPLACE (Note 3)
    uint8_t flags = combop;
    if (color)
        flags |= 0x08;
    d.push_back(flags);
    printf("region-info handler (%zu bytes, colored=%d combop=%u)\n", d.size(), color, combop);
    return { d, color, combop, width, height, x, y };
}

// 7.4.8: page information data part.
// Called after this page's content segments have already been generated
// (main() reserves the page-info segment number up front, per 7.4.8's "must
// be the first segment associated with the page," but builds its bytes
// last and splices them back to that position — see main()). That means
// `prior` already contains every content segment this page will have, so
// the content-dependent flags below can be computed exactly instead of
// guessed.
SegResult gen_segment_page_info(const std::vector<GeneratedSegment> &prior)
{
    std::vector<uint8_t> d;
    // 7.4.8.1/.2: page width and height -- drawn by choose_page_geometry()
    // before any content segment (see g_page_width), not here, because
    // region placement has to be able to aim at a page that already has a
    // size. This segment only reports them.
    put_be32(d, g_page_width);
    put_be32(d, g_page_height);
    put_be32(d, 1 + (urand() % 300));         // x resolution
    put_be32(d, 1 + (urand() % 300));         // y resolution

    // Derive the content-dependent bits from this page's actual region
    // segments (8.2, 7.4.7.5, 7.4.1.5 Note 1).
    bool has_refinement = false, has_aux_buffer = false, has_colored = false;
    bool combop_overridden = false, combop_seen = false;
    uint8_t page_combop = (uint8_t)(urand() % 4);
    for (const auto &seg : prior) {
        if (seg.page != 1)
            continue;
        bool is_intermediate = seg.type == SEG_INTERMEDIATE_TEXT || seg.type == SEG_INTERMEDIATE_HALFTONE ||
                                seg.type == SEG_INTERMEDIATE_GENERIC || seg.type == SEG_INTERMEDIATE_GENERIC_REFINEMENT;
        bool is_refinement = seg.type == SEG_INTERMEDIATE_GENERIC_REFINEMENT ||
                              seg.type == SEG_IMMEDIATE_GENERIC_REFINEMENT ||
                              seg.type == SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT;
        // "Direct" region segments are combined straight into the page
        // buffer using their own external combination operator (8.2 step
        // 5a/5c/5d); intermediate segments are drawn into an auxiliary
        // buffer instead and their combop isn't used for the page (Note 2).
        bool is_direct = seg.type == SEG_IMMEDIATE_TEXT || seg.type == SEG_IMMEDIATE_LOSSLESS_TEXT ||
                          seg.type == SEG_IMMEDIATE_HALFTONE || seg.type == SEG_IMMEDIATE_LOSSLESS_HALFTONE ||
                          seg.type == SEG_IMMEDIATE_GENERIC || seg.type == SEG_IMMEDIATE_LOSSLESS_GENERIC ||
                          seg.type == SEG_IMMEDIATE_GENERIC_REFINEMENT || seg.type == SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT;

        if (is_intermediate)
            has_aux_buffer = true;             // 8.2 step 5b
        if (is_refinement) {
            has_refinement = true;
            if (seg.nrefs > 0)
                has_aux_buffer = true;         // 8.2 step 5d/5e: refines another segment's auxiliary buffer
        }
        if (seg.colored)
            has_colored = true;
        if (is_direct && seg.combop >= 0) {
            // 7.4.8.5 gives the page default combination operator only
            // bits 3-4, so it spans 0-3 (OR/AND/XOR/XNOR) -- REPLACE (4)
            // is legal as a *region's* external combination operator
            // (7.4.1.5) but has no encoding as a page default. Adopting it
            // here would shift 4 into bit 5 and forge the "requires
            // auxiliary buffers" flag. A region whose operator can't be
            // (or simply isn't) the page default is exactly what bit 6,
            // "combination operator overridden", exists to announce.
            if (seg.combop <= 3 && !combop_seen) {
                page_combop = (uint8_t)seg.combop;
                combop_seen = true;
            } else if ((uint8_t)seg.combop != page_combop) {
                combop_overridden = true;
            }
        }
    }

    // 7.4.8.5 page segment flags: bit 0 lossless, bit 1 might contain
    // refinements, bit 2 default pixel value, bits 3-4 default combination
    // operator (0-3: OR/AND/XOR/XNOR), bit 5 requires auxiliary buffers,
    // bit 6 combination operator overridden, bit 7 might contain coloured
    // segment. Bits 1, 5, 6, 7 and the combop in bits 3-4 are derived from
    // this page's actual content above; bit 0 has no content-level rule to
    // derive from, so it's left random. Bit 2 reports the default pixel
    // value choose_page_geometry() already made -- the page buffer is
    // filled with it before any region composes onto it, so it is part of
    // the page state g_page_bitmap models, not a free choice here.
    uint8_t flags = 0;
    if (urand() & 1)
        flags |= 0x01;
    if (has_refinement)
        flags |= 0x02;
    if (g_page_default_pixel)
        flags |= 0x04;
    if (page_combop > 3) {   // would overflow bits 3-4 into "requires auxiliary buffers"
        fprintf(stderr, "gen_segment_page_info: page default combop %u exceeds bits 3-4\n", page_combop);
        exit(1);
    }
    flags |= (uint8_t)(page_combop << 3);
    if (has_aux_buffer)
        flags |= 0x20;
    if (combop_overridden)
        flags |= 0x40;
    if (has_colored)
        flags |= 0x80;
    d.push_back(flags);

    // 7.4.8.6 page striping information (2 bytes): bit 15 page is striped,
    // bits 0-14 maximum stripe size.
    bool striped = (urand() & 1) != 0;
    uint16_t striping = (uint16_t)(urand() % 0x8000);
    if (striped)
        striping |= 0x8000;
    put_be16(d, striping);

    printf("page-info handler (%zu bytes, refinement=%d aux=%d colored=%d combop_overridden=%d combop=%u)\n",
           d.size(), has_refinement, has_aux_buffer, has_colored, combop_overridden, page_combop);
    return { d, {} };
}

// 7.4.15.1/.2: a short printable ISO/IEC 8859-1 string for one half of a
// comment's (name, value) pair -- no embedded 0x00, since that's the pair
// element's own terminator.
static std::string rand_comment_string(void)
{
    int len = 1 + (int)(urand() % 12);
    std::string s;
    s.reserve((size_t)len);
    for (int i = 0; i < len; i++)
        s.push_back((char)(0x20 + (urand() % (0x7F - 0x20))));   // printable ASCII
    return s;
}

// 7.4.14: extension segment data begins with a 4-byte extension type field.
// Bit 29 reserved, bit 30 dependent, bit 31 necessary; if necessary is set,
// reserved must also be set (7.4.14). pdfium ignores extension segment
// content entirely regardless of type (jbig2_context.cpp's case 62 just
// skips the declared data length), so unlike every other segment here,
// there's no decoder-verifiable signal for getting 7.4.15's payload byte
// format right -- this just matches the spec text itself.
SegResult gen_segment_extension(const std::vector<GeneratedSegment> &prior)
{
    std::vector<uint8_t> d;

    // 0: an undefined/reserved type (opaque, no defined payload format).
    // 1: 7.4.15.1 single-byte coded comment. 2: 7.4.15.2 multi-byte coded
    // comment (UCS-2).
    int kind = (int)(urand() % 3);
    uint32_t type = kind == 1 ? 0x20000000u : kind == 2 ? 0x20000002u
                                                          : (urand() & 0x1FFFFFFFu);
    if (urand() & 1)
        type |= 0x40000000u;               // bit 30: dependent
    if (urand() & 1) {
        type |= 0x80000000u;               // bit 31: necessary
        type |= 0x20000000u;               // bit 29: reserved, forced 1
    } else if (urand() & 1) {
        type |= 0x20000000u;               // bit 29: reserved, set independently
    }
    put_be32(d, type);

    // 7.4.15.1/.2: a number of (name, value) pairs, each element a string
    // terminated by 0x00 (single-byte) or 0x0000 (multi-byte), with one
    // more terminator after the last pair.
    if (kind == 1 || kind == 2) {
        int npairs = 1 + (int)(urand() % 3);
        for (int i = 0; i < npairs; i++) {
            for (int half = 0; half < 2; half++) {
                std::string s = rand_comment_string();
                if (kind == 1) {
                    append(d, (const uint8_t *)s.data(), s.size());
                    d.push_back(0x00);
                } else {
                    for (char c : s) {
                        d.push_back(0x00);          // UCS-2, BMP ASCII range: high byte 0
                        d.push_back((uint8_t)c);
                    }
                    d.push_back(0x00);
                    d.push_back(0x00);
                }
            }
        }
        d.push_back(0x00);
        if (kind == 2)
            d.push_back(0x00);
    }

    // 7.4.15.1: a comment extension may apply to specific referred-to
    // segments; pick up to two distinct prior segments, of any type.
    std::vector<uint32_t> pool;
    for (const auto &seg : prior)
        pool.push_back(seg.number);
    std::vector<uint32_t> refs = pick_refs(pool, 0, 2);   // any type, and optional

    printf("extension handler (%zu bytes, %zu refs%s)\n", d.size(), refs.size(),
           kind == 1 ? ", single-byte comment" : kind == 2 ? ", multi-byte comment" : "");
    return { d, refs };
}

// 7.4.4.1: pattern dictionary segment data header (Figure 41).
SegResult gen_segment_pattern_dict(const std::vector<GeneratedSegment> &)
{
    std::vector<uint8_t> d;
    bool mmr = (urand() & 1) != 0;
    // 7.4.4.1.1 flag: bit 0 HDMMR, bits 1-2 HDTEMPLATE; HDTEMPLATE must be
    // 0 when HDMMR is set. Bits 3-7 reserved, always 0.
    uint8_t flags = mmr ? 0x01 : 0x00;
    if (!mmr)
        flags |= (uint8_t)((urand() % 4) << 1);
    d.push_back(flags);
    // HDPW must be > 0 (7.4.4.1.2), and, when HDMMR is 0, Table 27 fixes
    // this decode's AT1 pixel at (-HDPW, 0); Figure 7 restricts that
    // pixel's X to [-128,-1] when Y is 0, so HDPW can be at most 128. Either
    // way the collective bitmap is real content (MMR below, or arithmetic
    // via CJBig2_PDDProc::DecodeArith's fixed-AT generic-region decode --
    // jbig2_pdd_proc.cpp:22-34), so HDPW/HDPH/GRAYMAX stay small enough to
    // keep its size ((GRAYMAX+1)*HDPW x HDPH) manageable to generate and
    // encode either way.
    uint32_t hdpw = 1 + urand() % 16;
    uint32_t hdph = 1 + urand() % 16;
    uint32_t graymax = urand() % 4;
    d.push_back((uint8_t)hdpw);    // HDPW
    d.push_back((uint8_t)hdph);    // HDPH, must be > 0
    put_be32(d, graymax);          // GRAYMAX = npatterns - 1

    uint32_t npatterns = graymax + 1;
    uint32_t totwidth = npatterns * hdpw;
    std::vector<uint8_t> px((size_t)totwidth * hdph);
    for (uint32_t y = 0; y < hdph; y++)
        for (uint32_t x = 0; x < totwidth; x++)
            px[(size_t)y * totwidth + x] = (uint8_t)((x ^ y) & 1);

    if (mmr) {
        // 6.7.5: real MMR-coded collective bitmap, (GRAYMAX+1) patterns of
        // HDPW x HDPH concatenated left to right, genuinely decodable via
        // T.6 (6.2.6: known length, from the segment's own data length).
        std::vector<uint8_t> coded = mmr_encode((int)totwidth, (int)hdph, px.data());
        append(d, coded.data(), coded.size());
    } else {
        // 6.7.5 + Table 27: the collective bitmap is decoded via the
        // generic region decoding procedure with TPGDON off and AT pixels
        // *fixed* (not signalled in the bitstream at all, unlike a generic
        // region segment) -- AT1 = (-HDPW, 0) always, and for HDTEMPLATE 0
        // the other three pinned at the same nominal values
        // write_nominal_at_pixel() uses (jbig2_pdd_proc.cpp:22-34, verified
        // against the real decoder's exact hardcoding). Reuses the same
        // template0/123 encoders every other real generic-region content
        // does; the unknown-length trailer generic regions need doesn't
        // apply here since this is a plain length-prefixed segment.
        int8_t at1x = (int8_t)(-(int32_t)hdpw), at1y = 0;
        std::vector<uint8_t> coded =
            (flags & 0x06) == 0   // HDTEMPLATE (bits 1-2) == 0
                ? mq_encode_generic_template0((int)totwidth, (int)hdph, px.data(),
                                               at1x, at1y, -3, -1, 2, -2, -2, -2, false)
                : mq_encode_generic_template123((int)totwidth, (int)hdph, px.data(),
                                                 (flags >> 1) & 0x03, at1x, at1y, false);
        append(d, coded.data(), coded.size());
    }

    printf("pattern-dictionary handler (%zu bytes, real %s content)\n",
           d.size(), mmr ? "MMR" : "arithmetic");
    SegResult r;
    r.data = d;
    // The whole collective bitmap, so a halftone-region generator wants
    // real patterns to select from: pattern i is the hdpw x hdph slice at
    // column i*hdpw (mirrors CJBig2_PDDProc::DecodeArith's own
    // SubImage(HDPW*GRAY, 0, HDPW, HDPH) split).
    r.bw = totwidth;
    r.bh = hdph;
    r.bitmap = std::move(px);
    r.hdpw = hdpw;
    return r;
}

// 7.4.2.1: symbol dictionary segment data header (Figure 32).
// A small synthetic glyph bitmap. The exact pixels don't matter -- only
// that a real decoder can decode the bytes back into a WxH bitmap -- so a
// deterministic diagonal-stripe pattern is as good as any.
struct Glyph {
    int w, h;
    std::vector<bool> px;   // row-major, px[y*w+x]
};

static Glyph make_glyph(int w, int h)
{
    Glyph g;
    g.w = w;
    g.h = h;
    g.px.resize((size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            g.px[(size_t)y * w + x] = ((x ^ y) & 1) != 0;
    return g;
}

// 6.5.9: packs a height class's symbols into one MSB-first, row-major,
// byte-per-row-padded collective bitmap (height x totwidth), symbols
// concatenated left to right.
static void pack_collective_bitmap(std::vector<uint8_t> &d, int height, int totwidth,
                                    const std::vector<Glyph> &syms)
{
    int stride = (totwidth + 7) / 8;
    std::vector<uint8_t> rows((size_t)stride * height, 0);
    int xoff = 0;
    for (const Glyph &g : syms) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < g.w; x++) {
                if (g.px[(size_t)y * g.w + x]) {
                    int gx = xoff + x;
                    rows[(size_t)y * stride + gx / 8] |= (uint8_t)(0x80 >> (gx % 8));
                }
            }
        }
        xoff += g.w;
    }
    append(d, rows.data(), rows.size());
}

// 7.4.2.1 + 6.5: a real, decodable Huffman-coded (SDHUFF=1) symbol
// dictionary with no refinement/aggregation (SDREFAGG=0), used by
// gen_segment_symbol_dict() whenever that's the combination it randomly
// picks. Standard tables only (no user-supplied tables, so no
// tables-segment references needed) and no imported symbols
// (SDNUMINSYMS=0), to keep this self-contained: heights strictly increase
// across height classes and widths strictly increase within one, so every
// DH/DW delta is positive and Table B.2's (no-negative-range) encoding is
// always valid regardless of which standard table gets picked.
// Height-class collective bitmaps are stored uncompressed (BMSIZE=0,
// 6.5.9 step 3) rather than MMR-coded.
SegResult gen_symbol_dict_real(void)
{
    std::vector<uint8_t> d;

    bool dh_table_b5 = (urand() & 1) != 0;   // false=B.4, true=B.5
    bool dw_table_b3 = (urand() & 1) != 0;   // false=B.2, true=B.3

    // 7.4.2.1.1: symbol dictionary flags. SDHUFF=1 (bit 0), SDREFAGG=0
    // (bit 1); SDHUFFDH/SDHUFFDW select between the two standard tables
    // for each (bits 2-3, 4-5); SDHUFFBMSIZE stays 0 = Table B.1 (bit 6);
    // SDHUFFAGGINST/context bits/SDTEMPLATE/SDRTEMPLATE are all "must be
    // 0" here since SDREFAGG=0 and SDHUFF=1.
    uint16_t flags = 0x0001;
    flags |= (uint16_t)((dh_table_b5 ? 1 : 0) << 2);
    flags |= (uint16_t)((dw_table_b3 ? 1 : 0) << 4);
    put_be16(d, flags);

    // Build 1-2 height classes of 1-3 small synthetic glyphs each, heights
    // and (within a class) widths strictly increasing.
    int nclasses = 1 + (int)(urand() % 2);
    std::vector<std::vector<Glyph>> classes;
    std::vector<int> class_height;
    uint32_t total_syms = 0;
    int prev_height = 0;
    for (int c = 0; c < nclasses; c++) {
        int height = prev_height + 1 + (int)(urand() % 8);
        prev_height = height;
        class_height.push_back(height);
        int nsyms = 1 + (int)(urand() % 3);
        std::vector<Glyph> syms;
        int prev_width = 0;
        for (int i = 0; i < nsyms; i++) {
            int width = prev_width + 1 + (int)(urand() % 8);
            prev_width = width;
            syms.push_back(make_glyph(width, height));
            total_syms++;
        }
        classes.push_back(std::move(syms));
    }

    put_be32(d, total_syms);   // 7.4.2.1.4: SDNUMEXSYMS -- export everything
    put_be32(d, total_syms);   // 7.4.2.1.5: SDNUMNEWSYMS

    // 6.5.5/6.5.6/6.5.7/6.5.9: each height class is HCDH, then DW per
    // symbol terminated by OOB, then the byte-aligned BMSIZE=0 + raw
    // collective bitmap. The Huffman-coded region and the raw bitmap
    // bytes alternate, so the bit writer is flushed to `d` and restarted
    // around every collective bitmap.
    BitWriter bw;
    StdHuffTable dh_table = dh_table_b5 ? STD_TABLE(HUFF_B5) : STD_TABLE(HUFF_B4);
    StdHuffTable dw_table = dw_table_b3 ? STD_TABLE(HUFF_B3) : STD_TABLE(HUFF_B2);
    prev_height = 0;
    for (int c = 0; c < nclasses; c++) {
        int hcdh = class_height[c] - prev_height;
        prev_height = class_height[c];
        huff_encode(bw, dh_table, hcdh);

        int totwidth = 0;
        int prev_width = 0;
        for (const Glyph &g : classes[c]) {
            huff_encode(bw, dw_table, g.w - prev_width);
            prev_width = g.w;
            totwidth += g.w;
        }
        huff_encode(bw, dw_table, OOB_VAL);   // end of height class

        huff_encode(bw, STD_TABLE(HUFF_B1), 0);   // BMSIZE = 0: collective bitmap is uncompressed
        bw_finish(bw);
        append(d, bw.bytes.data(), bw.bytes.size());
        bw = BitWriter();

        pack_collective_bitmap(d, class_height[c], totwidth, classes[c]);
    }

    // 6.5.10: export every symbol via two runs (not-exported length 0,
    // then exported length total_syms), both Table B.1-coded.
    huff_encode(bw, STD_TABLE(HUFF_B1), 0);
    huff_encode(bw, STD_TABLE(HUFF_B1), (int32_t)total_syms);
    bw_finish(bw);
    append(d, bw.bytes.data(), bw.bytes.size());

    printf("symbol-dictionary handler (%zu bytes, %u symbols in %d height classes, real content)\n",
           d.size(), total_syms, nclasses);
    SegResult r;
    r.data = d;
    r.num_symbols = total_syms;
    // 7.4.3.1.6: a decoder assigns SBSYMS in referred-to-dictionary order,
    // then within a dictionary in the order its own new symbols were
    // decoded -- exactly `classes` in class order, then within-class order,
    // since SDNUMINSYMS is always 0 here (7.4.2.2, no imported symbols) and
    // "export everything" above means every decoded symbol is exported.
    for (const auto &syms : classes) {
        for (const Glyph &g : syms) {
            ExportedSymbol es;
            es.w = (uint32_t)g.w;
            es.h = (uint32_t)g.h;
            es.px.resize(g.px.size());
            for (size_t i = 0; i < g.px.size(); i++)
                es.px[i] = g.px[i] ? 1 : 0;
            r.symbols.push_back(std::move(es));
        }
    }
    return r;
}

// 7.4.2.1 + 6.5: a real, decodable arithmetic-coded (SDHUFF=0) symbol
// dictionary with no refinement/aggregation (SDREFAGG=0). Same shape as
// gen_symbol_dict_real() (1-2 height classes of small synthetic glyphs,
// SDNUMINSYMS=0, export everything) but every structural integer (HCDH,
// DW, the export runs) goes through the Annex A.2 arithmetic integer
// procedure instead of a Huffman table, and each symbol's bitmap is
// arithmetic generic-region content (mq_encode_generic_template*) instead
// of a packed collective bitmap. CJBig2_SDDProc::DecodeArith
// (jbig2_sdd_proc.cpp) runs one continuous arithmetic-coded stream and one
// shared gbContexts array across HCDH/DW/every symbol bitmap/the export
// runs alike -- no per-field or per-symbol reset -- so this uses one
// MQEncoder and one generic-region context array throughout, flushed once
// at the very end, the same pattern gen_segment_halftone_region() needs
// for its own multi-part continuous stream. This is the one decode path
// (CJBig2_GRDProc::DecodeArith, as opposed to the Progressive/
// StartDecodeArith family every standalone generic region segment uses)
// no other real content in this generator reaches -- see the
// TODO(coverage) comment on gen_segment_symbol_dict()'s structural
// fallback, which this closes.
SegResult gen_symbol_dict_real_arith(void)
{
    std::vector<uint8_t> d;

    uint8_t sdtemplate = (uint8_t)(urand() % 4);

    // 7.4.2.1.1: SDHUFF=0 (bit 0), SDREFAGG=0 (bit 1); SDTEMPLATE (bits
    // 10-11); everything else (SDHUFFxx selectors, context-reuse bits,
    // SDRTEMPLATE) is meaningless when SDHUFF=0/SDREFAGG=0, stays 0.
    uint16_t flags = (uint16_t)(sdtemplate << 10);
    put_be16(d, flags);

    // 7.4.2.1.2: symbol dictionary AT flags, present since SDHUFF == 0.
    int npairs = sdtemplate == 0 ? 4 : 1;
    AtPixel at[4] = {};
    for (int i = 0; i < npairs; i++)
        at[i] = write_primary_at_pixel(d);   // SDATXn/SDATYn

    // Build 1-2 height classes of 1-3 small synthetic glyphs each, same as
    // gen_symbol_dict_real() -- heights and (within a class) widths need
    // not be positive here (IAEX/IADH/IADW have no Table-B.2/B.4-style
    // "no negative range" restriction the way those standard Huffman
    // tables do), but keeping them strictly increasing still keeps the
    // generated bitmaps varied without needing negative deltas at all.
    int nclasses = 1 + (int)(urand() % 2);
    std::vector<std::vector<Glyph>> classes;
    std::vector<int> class_height;
    uint32_t total_syms = 0;
    int prev_height = 0;
    for (int c = 0; c < nclasses; c++) {
        int height = prev_height + 1 + (int)(urand() % 8);
        prev_height = height;
        class_height.push_back(height);
        int nsyms = 1 + (int)(urand() % 3);
        std::vector<Glyph> syms;
        int prev_width = 0;
        for (int i = 0; i < nsyms; i++) {
            int width = prev_width + 1 + (int)(urand() % 8);
            prev_width = width;
            syms.push_back(make_glyph(width, height));
            total_syms++;
        }
        classes.push_back(std::move(syms));
    }

    put_be32(d, total_syms);   // 7.4.2.1.4: SDNUMEXSYMS -- export everything
    put_be32(d, total_syms);   // 7.4.2.1.5: SDNUMNEWSYMS

    // One continuous arithmetic-coded stream: IADH/IADW/IAEX each get their
    // own persistent 512-entry context (Annex A.2), and every symbol
    // bitmap shares one gbContexts array sized for SDTEMPLATE -- matching
    // CJBig2_SDDProc::DecodeArith's single `gbContexts` span reused across
    // every pGRD->DecodeArith() call in the whole segment.
    MQEncoder mq;
    ArithIntCtx iadh, iadw, iaex;
    std::vector<uint8_t> gbcx(1u << (sdtemplate == 0 ? 16 : sdtemplate == 1 ? 13 : 10), 0);
    int atx1 = at[0].x, aty1 = at[0].y;

    prev_height = 0;
    for (int c = 0; c < nclasses; c++) {
        int hcdh = class_height[c] - prev_height;
        prev_height = class_height[c];
        mq_encode_arith_int(mq, iadh, hcdh);

        int prev_width = 0;
        for (const Glyph &g : classes[c]) {
            mq_encode_arith_int(mq, iadw, g.w - prev_width);
            prev_width = g.w;

            std::vector<uint8_t> px((size_t)g.w * g.h);
            for (size_t i = 0; i < g.px.size(); i++)
                px[i] = g.px[i] ? 1 : 0;
            if (sdtemplate == 0)
                mq_encode_generic_template0_into(mq, gbcx, g.w, g.h, px.data(),
                                                  atx1, aty1, at[1].x, at[1].y,
                                                  at[2].x, at[2].y, at[3].x, at[3].y, false);
            else
                mq_encode_generic_template123_into(mq, gbcx, g.w, g.h, px.data(),
                                                    sdtemplate, atx1, aty1, false);
        }
        mq_encode_arith_int(mq, iadw, 0, /*oob=*/true);   // end of height class
    }

    // 6.5.10: export every symbol via two runs (not-exported length 0,
    // then exported length total_syms) -- same shape as
    // gen_symbol_dict_real()'s Huffman version, via IAEX instead of Table
    // B.1.
    mq_encode_arith_int(mq, iaex, 0);
    mq_encode_arith_int(mq, iaex, (int32_t)total_syms);
    mq_flush(mq);
    append(d, mq.out.data(), mq.out.size());
    // No RSIZE-style length field constrains this stream's exact byte
    // count (unlike the Huffman-embedded refinement cases elsewhere in
    // this generator), so a plain trailing margin is enough -- the same
    // pattern gen_segment_generic_region()'s and gen_segment_pattern_dict()'s
    // own real arithmetic content already use.
    d.push_back(0xFF);
    d.push_back(0xFF);

    printf("symbol-dictionary handler (%zu bytes, %u symbols in %d height classes, real arithmetic content)\n",
           d.size(), total_syms, nclasses);
    SegResult r;
    r.data = d;
    r.num_symbols = total_syms;
    for (const auto &syms : classes) {
        for (const Glyph &g : syms) {
            ExportedSymbol es;
            es.w = (uint32_t)g.w;
            es.h = (uint32_t)g.h;
            es.px.resize(g.px.size());
            for (size_t i = 0; i < g.px.size(); i++)
                es.px[i] = g.px[i] ? 1 : 0;
            r.symbols.push_back(std::move(es));
        }
    }
    return r;
}

// 7.4.2.1 + 6.5.8.2.2: a real, decodable Huffman-coded (SDHUFF=1) symbol
// dictionary with refinement/aggregate coding (SDREFAGG=1), every new
// symbol using REFAGGNINST=1 -- REFAGGNINST>1 pulls in a full nested
// text-region decode (6.5.8.2.1) that isn't modeled here, so this always
// codes exactly the one value a decoder can take that "simple" path with.
// Once SDREFAGG=1, there is no "plain new bitmap" option at all -- 6.5.8.1
// always drives every new symbol through the refinement/aggregate coding
// procedure -- so the very first symbol has nothing to refine against
// except an *imported* one; the caller only takes this path when `prior`
// has a real dictionary to import from. Importing pulls in that whole
// dictionary's export list (7.4.2.2), so SDNUMINSYMS is its full export
// count, not 1 -- IDI is always 0 (the first imported symbol), but
// SBSYMCODELEN (IDI's bit width) still has to be computed from the real
// SDNUMINSYMS, same as a decoder would. SYMWIDTH/HCHEIGHT are coded to
// equal that imported symbol's own
// size exactly, so RDXI=RDYI=0 refines it against itself at no offset --
// mirroring gen_segment_refinement_region()'s and gen_text_region_real()'s
// same-size real-reference cases, and reusing their mq_encode_refinement()/
// mq_finalize_refinement() machinery unchanged.
SegResult gen_symbol_dict_real_refagg(const std::vector<GeneratedSegment> &prior, uint32_t import_from)
{
    // Importing a dictionary imports *every* symbol it exports (7.4.2.2),
    // so SDNUMINSYMS is that dictionary's whole export count, not 1 -- only
    // the first imported symbol (index 0) is actually used as the seed
    // reference, but SDNUMINSYMS still has to be the true count for
    // SBSYMCODELEN (IDI's bit width, jbig2_sdd_proc.cpp:350-354) and the
    // export run-lengths below to come out right.
    const ExportedSymbol *seed = nullptr;
    uint32_t sdnuminsyms = 0;
    for (const auto &seg : prior)
        if (seg.number == import_from && !seg.symbols.empty()) {
            seed = &seg.symbols[0];
            sdnuminsyms = seg.num_symbols;
            break;
        }
    uint32_t sbsymcodelen = 1;
    while ((1u << sbsymcodelen) < sdnuminsyms + 1)
        sbsymcodelen++;

    std::vector<uint8_t> d;

    // GRTEMPLATE == 1 refinement here always hits pdfium's
    // DecodeTemplate1Opt (GRW == SYMWIDTH == seed->w == GRREFERENCE->width()
    // and GRREFERENCEDX == RDXI == 0 always hold, jbig2_sdd_proc.cpp's
    // REFAGGNINST==1 branch). That routine reads the reference bitmap a
    // whole byte at a time without masking off the bits beyond the symbol's
    // real width in a row's last byte -- see gen_text_region_real_arith()'s
    // identical guard for the full explanation. Only allow SDRTEMPLATE == 1
    // when the seed's width is byte-aligned, so no row has a partial last
    // byte for Opt to misread.
    uint8_t sdrtemplate = (seed->w % 8 == 0) ? (uint8_t)(urand() % 2) : 0;

    // 7.4.2.1.1: SDHUFF=1 (bit 0), SDREFAGG=1 (bit 1); SDHUFFDH/SDHUFFDW
    // select standard tables B.4/B.2 (bits 2-3, 4-5 left 0) -- unused for
    // BMSIZE/the collective bitmap (SDREFAGG=1 skips both, 6.5.8.1) but
    // still spent on HCDH/DW below; SDHUFFBMSIZE/SDHUFFAGGINST both stay 0
    // (standard tables B.1) -- the REFAGGNINST==1 branch hardcodes its own
    // local B.15/B.15/B.1 tables for RDX/RDY/BMSIZE regardless of these
    // bits (jbig2_sdd_proc.cpp:374-375), so only SDHUFFAGGINST's Table B.1
    // read for REFAGGNINST itself actually depends on this selector.
    uint16_t flags = 0x0001 | 0x0002;
    flags |= (uint16_t)(sdrtemplate << 12);
    put_be16(d, flags);

    // 7.4.2.1.2: symbol dictionary AT flags -- absent, SDHUFF == 1.
    // 7.4.2.1.3: refinement AT flags, present since SDREFAGG == 1; only
    // when SDRTEMPLATE == 0.
    AtPixel at1 = {}, at2 = {};
    if (sdrtemplate == 0) {
        at1 = write_primary_at_pixel(d);       // SDRATX1/SDRATY1
        at2 = write_reference_at_pixel(d);     // SDRATX2/SDRATY2
    }

    put_be32(d, 1);   // 7.4.2.1.4: SDNUMEXSYMS -- export the one new symbol
    put_be32(d, 1);   // 7.4.2.1.5: SDNUMNEWSYMS

    // Coarse blocks, not a copy of the reference, give the refinement coder
    // genuine new content to code against a real reference -- the same
    // choice gen_text_region_real()'s SBREFINE=1 path makes. Computed before
    // any bits are written below: RSIZE (the encoded byte count) has to be
    // known before it can be Huffman-coded into the bitstream, and this
    // computation is otherwise independent of stream position.
    std::vector<uint8_t> cur((size_t)seed->w * seed->h);
    for (uint32_t y = 0; y < seed->h; y++)
        for (uint32_t x = 0; x < seed->w; x++)
            cur[(size_t)y * seed->w + x] = (uint8_t)(((x >> 1) + (y >> 1)) & 1);
    std::vector<uint8_t> coded = mq_encode_refinement(
        (int)seed->w, (int)seed->h, cur.data(), seed->px.data(),
        sdrtemplate, /*tpgron=*/false, at1.x, at1.y, at2.x, at2.y);
    uint32_t rsize;
    std::vector<uint8_t> final_bytes = mq_finalize_refinement(
        (int)seed->w, (int)seed->h, seed->px.data(), sdrtemplate,
        at1.x, at1.y, at2.x, at2.y, coded, &rsize);

    // HCDH, DW, REFAGGNINST, IDI, RDXI, RDYI, and BMSIZE are all read back
    // to back off one continuous bitstream (jbig2_sdd_proc.cpp:349-385) --
    // *no* byte alignment between IDI and RDXI, unlike the flush right
    // after every other symbol/height-class boundary elsewhere in this
    // generator -- only the arithmetic bytes that follow BMSIZE get their
    // own alignment (immediately below).
    BitWriter bw;
    huff_encode(bw, STD_TABLE(HUFF_B4), (int32_t)seed->h);   // HCDH: HCHEIGHT starts at 0
    huff_encode(bw, STD_TABLE(HUFF_B2), (int32_t)seed->w);   // DW: SYMWIDTH starts at 0
    huff_encode(bw, STD_TABLE(HUFF_B1), 1);                  // REFAGGNINST = 1
    // 6.5.8.2.2: IDI, SBSYMCODELEN raw bits -- 0, the first imported symbol.
    bw_put_bits(bw, 0, (int)sbsymcodelen);
    huff_encode(bw, STD_TABLE(HUFF_B15), 0);              // RDXI
    huff_encode(bw, STD_TABLE(HUFF_B15), 0);              // RDYI
    huff_encode(bw, STD_TABLE(HUFF_B1), (int32_t)rsize);  // BMSIZE
    bw_finish(bw);
    append(d, bw.bytes.data(), bw.bytes.size());
    bw = BitWriter();

    append(d, final_bytes.data(), final_bytes.size());
    d.push_back(0xFF);   // trailing bytes a decoder unconditionally skips
    d.push_back(0xFF);   // over -- see gen_text_region_real()'s comment

    // End the DW loop (6.5.7 step c) -- the only symbol in this height
    // class, so straight to OOB; SDREFAGG == 1 skips the BMSIZE/collective
    // -bitmap block entirely (jbig2_sdd_proc.cpp:418), so nothing else
    // follows before NSYMSDECODED (1) == SDNUMNEWSYMS (1) ends the outer
    // height-class loop too.
    huff_encode(bw, STD_TABLE(HUFF_B2), OOB_VAL);

    // 6.5.10: two runs -- not-exported (length SDNUMINSYMS, every imported
    // symbol) then exported (length 1, the new symbol at index SDNUMINSYMS)
    // -- covering SDNUMINSYMS + SDNUMNEWSYMS entries, both Table B.1-coded.
    huff_encode(bw, STD_TABLE(HUFF_B1), (int32_t)sdnuminsyms);
    huff_encode(bw, STD_TABLE(HUFF_B1), 1);
    bw_finish(bw);
    append(d, bw.bytes.data(), bw.bytes.size());

    printf("symbol-dictionary handler (%zu bytes, 1 symbol, real refinement/aggregate content)\n", d.size());
    SegResult r;
    r.data = d;
    r.refs = { import_from };
    r.num_symbols = 1;
    // mq_encode_refinement() rewrites `cur` in place to hold exactly what a
    // decoder reconstructs (tpgron is always false here, so unconditionally
    // exact -- same guarantee gen_text_region_real()'s refined instances
    // rely on).
    ExportedSymbol es;
    es.w = seed->w;
    es.h = seed->h;
    es.px = std::move(cur);
    r.symbols.push_back(std::move(es));
    return r;
}

// 7.4.2.1 + 6.5.8.2.2: a real, decodable arithmetic (SDHUFF=0) symbol
// dictionary with refinement/aggregate coding (SDREFAGG=1), REFAGGNINST=1 --
// the arithmetic-stream counterpart to gen_symbol_dict_real_refagg() above;
// see that function's comment for why REFAGGNINST is always 1, why the seed
// is always imported (never a same-segment new symbol), and why RDXI=RDYI=0
// (SYMWIDTH/HCHEIGHT coded to equal the seed's own size exactly). The only
// structural differences from the Huffman version: SDAT (symbol dictionary
// AT flags, 7.4.2.1.2) is present here regardless of SDREFAGG since it's
// gated on SDHUFF alone (jbig2_context.cpp:421-425) -- unused by the
// REFAGGNINST==1 decode path, but its bytes still have to be there for
// everything after them to land at the right offset -- and SBSYMCODELEN
// starts at 0 here (CJBig2_SDDProc::DecodeArith's own SBSYMCODELENA,
// jbig2_sdd_proc.cpp:40-43), unlike the Huffman refagg branch's
// starts-at-1 convention.
SegResult gen_symbol_dict_real_refagg_arith(const std::vector<GeneratedSegment> &prior, uint32_t import_from)
{
    const ExportedSymbol *seed = nullptr;
    uint32_t sdnuminsyms = 0;
    for (const auto &seg : prior)
        if (seg.number == import_from && !seg.symbols.empty()) {
            seed = &seg.symbols[0];
            sdnuminsyms = seg.num_symbols;
            break;
        }
    int sbsymcodelen = 0;
    while ((1u << sbsymcodelen) < sdnuminsyms + 1)
        sbsymcodelen++;

    std::vector<uint8_t> d;

    uint8_t sdtemplate = (uint8_t)(urand() % 4);   // unused by REFAGGNINST==1, but still selects SDAT's count
    // See gen_symbol_dict_real_refagg()'s identical guard: GRTEMPLATE == 1
    // here always hits pdfium's DecodeTemplate1Opt (GRW == seed->w ==
    // GRREFERENCE->width(), GRREFERENCEDX == RDXI == 0), which misreads
    // non-zero padding bits beyond a non-byte-aligned reference width.
    uint8_t sdrtemplate = (seed->w % 8 == 0) ? (uint8_t)(urand() % 2) : 0;

    // 7.4.2.1.1: SDHUFF=0 (bit 0), SDREFAGG=1 (bit 1); SDTEMPLATE (bits
    // 10-11, structurally present -- see the function comment -- but never
    // actually consulted since REFAGGNINST==1 never runs the plain generic-
    // region path); SDRTEMPLATE (bit 12).
    uint16_t flags = 0x0002;
    flags |= (uint16_t)(sdtemplate << 10);
    flags |= (uint16_t)((uint32_t)sdrtemplate << 12);
    put_be16(d, flags);

    // 7.4.2.1.2: symbol dictionary AT flags, present since SDHUFF == 0.
    int npairs = sdtemplate == 0 ? 4 : 1;
    for (int i = 0; i < npairs; i++)
        write_primary_at_pixel(d);   // SDATXn/SDATYn -- unused, see comment

    // 7.4.2.1.3: refinement AT flags, present since SDREFAGG == 1; only
    // when SDRTEMPLATE == 0.
    AtPixel at1 = {}, at2 = {};
    if (sdrtemplate == 0) {
        at1 = write_primary_at_pixel(d);       // SDRATX1/SDRATY1
        at2 = write_reference_at_pixel(d);     // SDRATX2/SDRATY2
    }

    put_be32(d, 1);   // 7.4.2.1.4: SDNUMEXSYMS -- export the one new symbol
    put_be32(d, 1);   // 7.4.2.1.5: SDNUMNEWSYMS

    // Coarse blocks, not a copy of the reference -- same choice
    // gen_symbol_dict_real_refagg() makes, see its comment.
    std::vector<uint8_t> cur((size_t)seed->w * seed->h);
    for (uint32_t y = 0; y < seed->h; y++)
        for (uint32_t x = 0; x < seed->w; x++)
            cur[(size_t)y * seed->w + x] = (uint8_t)(((x >> 1) + (y >> 1)) & 1);

    // One continuous arithmetic-coded stream (CJBig2_SDDProc::DecodeArith's
    // single pArithDecoder/gbContexts/grContexts spans, reused verbatim
    // across IADH/IADW/IAAI/IAID/IARDX/IARDY and the refinement decode
    // itself -- matching gen_symbol_dict_real_arith()'s and
    // gen_text_region_real_arith()'s identical one-stream shape).
    MQEncoder mq;
    ArithIntCtx iadh, iadw, iaai, iaex;
    ArithIntCtx iardx, iardy;
    std::vector<uint8_t> iaid_cx(1u << sbsymcodelen, 0);
    std::vector<uint8_t> refine_cx(1u << (sdrtemplate == 0 ? 13 : 10), 0);

    mq_encode_arith_int(mq, iadh, (int32_t)seed->h);   // HCDH: HCHEIGHT starts at 0
    mq_encode_arith_int(mq, iadw, (int32_t)seed->w);   // DW: SYMWIDTH starts at 0
    mq_encode_arith_int(mq, iaai, 1);                  // REFAGGNINST = 1
    mq_encode_arith_iaid(mq, iaid_cx, sbsymcodelen, 0);   // IDI = 0, the first imported symbol
    mq_encode_arith_int(mq, iardx, 0);                 // RDXI
    mq_encode_arith_int(mq, iardy, 0);                 // RDYI
    mq_encode_refinement_into(mq, refine_cx, (int)seed->w, (int)seed->h,
                               cur.data(), seed->px.data(),
                               sdrtemplate, /*tpgron=*/false,
                               at1.x, at1.y, at2.x, at2.y);

    // End the DW loop (6.5.7 step c) -- the only symbol in this height
    // class, so straight to OOB; NSYMSDECODED (1) == SDNUMNEWSYMS (1) then
    // ends the outer height-class loop too.
    mq_encode_arith_int(mq, iadw, 0, /*oob=*/true);

    // 6.5.10: two runs -- not-exported (length SDNUMINSYMS, every imported
    // symbol) then exported (length 1, the new symbol) -- via IAEX instead
    // of Table B.1.
    mq_encode_arith_int(mq, iaex, (int32_t)sdnuminsyms);
    mq_encode_arith_int(mq, iaex, 1);
    mq_flush(mq);
    append(d, mq.out.data(), mq.out.size());
    d.push_back(0xFF);
    d.push_back(0xFF);

    printf("symbol-dictionary handler (%zu bytes, 1 symbol, real arithmetic refinement/aggregate content)\n", d.size());
    SegResult r;
    r.data = d;
    r.refs = { import_from };
    r.num_symbols = 1;
    // tpgron is always false above, so mq_encode_refinement_into() never
    // touches `cur` -- it already holds exactly what a decoder reconstructs,
    // by construction.
    ExportedSymbol es;
    es.w = seed->w;
    es.h = seed->h;
    es.px = std::move(cur);
    r.symbols.push_back(std::move(es));
    return r;
}

// 7.4.2.1: symbol dictionary segment data header (Figure 32). SDHUFF and
// SDREFAGG are picked from the full range the spec allows; all four
// combinations now have real, decodable content: gen_symbol_dict_real()
// (SDHUFF=1/SDREFAGG=0) and gen_symbol_dict_real_arith() (SDHUFF=0/
// SDREFAGG=0) unconditionally, and gen_symbol_dict_real_refagg() (SDHUFF=1)
// / gen_symbol_dict_real_refagg_arith() (SDHUFF=0) whenever SDREFAGG=1 and
// `prior` has a real dictionary to import a seed symbol from. Without one,
// SDREFAGG=1 falls back to the structural path below (same "no real
// referent available" fallback every other real-content generator here
// uses).
SegResult gen_segment_symbol_dict(const std::vector<GeneratedSegment> &prior)
{
    bool sdhuff = (urand() & 1) != 0;
    bool sdrefagg = (urand() & 1) != 0;

    if (sdrefagg) {
        std::vector<uint32_t> real_dict_pool;
        for (const auto &seg : prior)
            if (seg.type == SEG_SYMBOL_DICTIONARY && seg.num_symbols > 0 && !seg.symbols.empty())
                real_dict_pool.push_back(seg.number);
        if (!real_dict_pool.empty()) {
            std::vector<uint32_t> chosen = pick_refs(real_dict_pool, 1, 1);
            return sdhuff ? gen_symbol_dict_real_refagg(prior, chosen[0])
                          : gen_symbol_dict_real_refagg_arith(prior, chosen[0]);
        }
    }

    if (sdhuff && !sdrefagg)
        return gen_symbol_dict_real();

    if (!sdhuff && !sdrefagg)
        return gen_symbol_dict_real_arith();

    std::vector<uint8_t> d;

    // Remaining structural fallback: SDREFAGG == 1 with no real dictionary
    // in `prior` to import a seed symbol from -- see gen_symbol_dict_real_
    // refagg()'s and gen_symbol_dict_real_refagg_arith()'s comments.
    //
    // 7.4.2.2: a symbol dictionary may import symbols from earlier symbol
    // dictionary segments.
    std::vector<uint32_t> dict_pool = segment_numbers_of_type(prior, SEG_SYMBOL_DICTIONARY);
    std::vector<uint32_t> refs = pick_refs(dict_pool, 0, 2);   // importing is optional

    uint8_t sdtemplate = sdhuff ? 0 : (uint8_t)(urand() % 4);
    uint8_t sdrtemplate = sdrefagg ? (uint8_t)(urand() % 2) : 0;

    // 7.4.2.1.6: a "user-supplied" (3) Huffman table selector must be
    // matched by a distinct referred-to tables segment, in the order
    // SDHUFFDH, SDHUFFDW, SDHUFFBMSIZE, SDHUFFAGGINST.
    std::vector<uint32_t> table_pool = segment_numbers_of_type(prior, SEG_TABLES);
    std::vector<uint32_t> table_refs;

    // 7.4.2.1.1: symbol dictionary flags (2 bytes).
    uint16_t flags = 0;
    if (sdhuff)
        flags |= 0x0001;
    if (sdrefagg)
        flags |= 0x0002;
    flags |= (uint16_t)((sdhuff ? pick_sel_013(table_pool, table_refs) : 0) << 2);   // SDHUFFDH
    flags |= (uint16_t)((sdhuff ? pick_sel_013(table_pool, table_refs) : 0) << 4);   // SDHUFFDW
    if (sdhuff && pick_sel_bit(table_pool, table_refs))
        flags |= 0x0040;                                        // SDHUFFBMSIZE
    if (sdhuff && sdrefagg && pick_sel_bit(table_pool, table_refs))
        flags |= 0x0080;                                        // SDHUFFAGGINST
    bool ctx_allowed = !(sdhuff && !sdrefagg);
    if (ctx_allowed && (urand() & 1))
        flags |= 0x0100;                                        // bitmap coding context used
    if (ctx_allowed && (urand() & 1))
        flags |= 0x0200;                                        // bitmap coding context retained
    flags |= (uint16_t)(sdtemplate << 10);
    flags |= (uint16_t)(sdrtemplate << 12);
    put_be16(d, flags);

    // 7.4.2.1.2: symbol dictionary AT flags, present only if SDHUFF == 0.
    if (!sdhuff) {
        int npairs = (sdtemplate == 0) ? 4 : 1;
        for (int i = 0; i < npairs; i++)
            write_primary_at_pixel(d);       // SDATXn/SDATYn
    }

    // 7.4.2.1.3: symbol dictionary refinement AT flags, present only if
    // SDREFAGG == 1 and SDRTEMPLATE == 0.
    if (sdrefagg && sdrtemplate == 0) {
        write_primary_at_pixel(d);           // SDRATX1/SDRATY1
        write_reference_at_pixel(d);         // SDRATX2/SDRATY2
    }

    put_be32(d, 1 + (urand() % 64));         // 7.4.2.1.4: SDNUMEXSYMS
    put_be32(d, 1 + (urand() % 64));         // 7.4.2.1.5: SDNUMNEWSYMS

    // Remainder: coded symbol bitmaps (and Huffman tables if SDHUFF), not modeled.
    append_random_payload(d, 256);

    for (uint32_t t : table_refs)
        refs.push_back(t);

    printf("symbol-dictionary handler (%zu bytes, %zu refs, structural)\n", d.size(), refs.size());
    return { d, refs };
}

// 7.4.3.1: text region segment data header (Figure 37). Starts with the
// common region segment information field (7.4.1).
// B.3's canonical code assignment, mirroring pdfium's
// CJBig2_Context::HuffmanAssignCode bit for bit: entries of each length get
// consecutive codes in index order, starting from
// firstcode[len] = (firstcode[len-1] + count[len-1]) << 1. A length of 0
// means "no code" -- the entry is skipped, and a decoder can never match it
// (its match test needs codelen == the number of bits read, always >= 1).
static void assign_canonical_codes(const std::vector<int> &lens, std::vector<uint32_t> &codes)
{
    codes.assign(lens.size(), 0);
    int lenmax = 0;
    for (int l : lens)
        if (l > lenmax)
            lenmax = l;
    std::vector<int> lencounts((size_t)lenmax + 1, 0), firstcodes((size_t)lenmax + 1, 0);
    for (int l : lens)
        lencounts[(size_t)l]++;
    lencounts[0] = 0;
    for (int i = 1; i <= lenmax; i++) {
        firstcodes[(size_t)i] = (firstcodes[(size_t)i - 1] + lencounts[(size_t)i - 1]) << 1;
        int cur = firstcodes[(size_t)i];
        for (size_t j = 0; j < lens.size(); j++)
            if (lens[j] == i)
                codes[j] = (uint32_t)cur++;
    }
}

// Code lengths for `weights.size()` entries, built by ordinary Huffman
// construction so they satisfy Kraft equality -- a *complete* prefix code,
// which is what keeps B.3's canonical assignment collision-free. Falls back
// to a uniform ceil(log2(n)) length if the tree comes out deeper than
// max_len (still collision-free, just not maximally compact), since both
// callers below have a hard ceiling on what they can express: a symbol ID
// code length is emitted as RUNCODE[length] and so cannot exceed 31, and a
// runcode's own length goes into a 4-bit field.
static std::vector<int> huffman_code_lengths(const std::vector<uint32_t> &weights, int max_len)
{
    size_t n = weights.size();
    std::vector<int> lens(n, 0);
    if (n == 0)
        return lens;
    if (n == 1) {
        lens[0] = 1;   // a lone entry still needs a real (1-bit) code
        return lens;
    }

    struct Node { uint64_t w; int left, right, leaf; };
    std::vector<Node> nodes;
    std::vector<int> live;
    for (size_t i = 0; i < n; i++) {
        nodes.push_back({ weights[i] ? weights[i] : 1, -1, -1, (int)i });
        live.push_back((int)i);
    }
    while (live.size() > 1) {
        size_t a = 0, b = 1;
        if (nodes[(size_t)live[b]].w < nodes[(size_t)live[a]].w)
            std::swap(a, b);
        for (size_t k = 2; k < live.size(); k++) {
            if (nodes[(size_t)live[k]].w < nodes[(size_t)live[a]].w) {
                b = a;
                a = k;
            } else if (nodes[(size_t)live[k]].w < nodes[(size_t)live[b]].w) {
                b = k;
            }
        }
        int ia = live[a], ib = live[b];
        nodes.push_back({ nodes[(size_t)ia].w + nodes[(size_t)ib].w, ia, ib, -1 });
        size_t hi = a > b ? a : b, lo = a > b ? b : a;
        live.erase(live.begin() + (ptrdiff_t)hi);
        live.erase(live.begin() + (ptrdiff_t)lo);
        live.push_back((int)nodes.size() - 1);
    }

    std::vector<std::pair<int, int>> stack{ { live[0], 0 } };
    while (!stack.empty()) {
        auto [idx, depth] = stack.back();
        stack.pop_back();
        if (nodes[(size_t)idx].leaf >= 0) {
            lens[(size_t)nodes[(size_t)idx].leaf] = depth < 1 ? 1 : depth;
            continue;
        }
        stack.push_back({ nodes[(size_t)idx].left, depth + 1 });
        stack.push_back({ nodes[(size_t)idx].right, depth + 1 });
    }

    int mx = 0;
    for (int l : lens)
        if (l > mx)
            mx = l;
    if (mx > max_len) {
        int L = 0;
        while (((size_t)1 << L) < n)
            L++;
        for (int &l : lens)
            l = L < 1 ? 1 : L;
    }
    return lens;
}

// The symbol ID Huffman table a real text region hands its decoder, plus
// the list of symbols it is actually allowed to place (7.4.3.1.7 lets a
// symbol carry code length 0, meaning "no code" -- such a symbol exists in
// SBSYMS but can never be selected as an IDI).
struct SymbolIDTable {
    std::vector<int> len;
    std::vector<uint32_t> code;
    std::vector<uint32_t> usable;
};

// 7.4.3.1.5/.1.7: the text region's symbol ID Huffman decoding table.
// SBNUMSYMS code lengths, themselves run-length coded with the RUNCODE
// alphabet of Table 32 and Huffman coded on top of that, preceded by the
// 35 four-bit RUNCODE lengths. Emitting genuinely varied lengths (rather
// than giving every symbol the same one, which collapses the table to a
// single active runcode and a run of identical 1-bit codes) is what makes
// a decoder actually run B.3's canonical assignment over both alphabets
// and take Table 32's repeat forms:
//   RUNCODE32  repeat the previous length 3-6 times   (2 extra bits)
//   RUNCODE33  repeat length 0 for 3-10 symbols       (3 extra bits)
//   RUNCODE34  repeat length 0 for 11-138 symbols     (7 extra bits)
// The field is byte-aligned at the end (step 6) so instance coding starts
// on a byte boundary.
static SymbolIDTable write_symbol_id_table(std::vector<uint8_t> &d, uint32_t sbnumsyms)
{
    SymbolIDTable t;
    t.len.assign(sbnumsyms, 0);

    if (sbnumsyms > 0) {
        // Leave some symbols unusable (code length 0) so runs of zeros --
        // and with them RUNCODE33/34 -- actually arise. A single wide gap
        // reaches the longer forms that scattered single zeros never would.
        std::vector<bool> used(sbnumsyms, true);
        if (sbnumsyms >= 4 && (urand() & 1)) {
            uint32_t runlen = 3 + urand() % (sbnumsyms - 3);
            uint32_t start = urand() % (sbnumsyms - runlen + 1);
            for (uint32_t i = 0; i < runlen; i++)
                used[start + i] = false;
        } else {
            for (uint32_t i = 0; i < sbnumsyms; i++)
                used[i] = (urand() % 4) != 0;
        }
        std::vector<uint32_t> weights;
        for (uint32_t i = 0; i < sbnumsyms; i++)
            if (used[i]) {
                t.usable.push_back(i);
                weights.push_back(1 + urand() % 8);
            }
        if (t.usable.empty()) {   // every symbol drawn unusable; keep one
            uint32_t keep = urand() % sbnumsyms;
            t.usable.push_back(keep);
            weights.push_back(1);
        }
        std::vector<int> ulens = huffman_code_lengths(weights, 15);
        for (size_t i = 0; i < t.usable.size(); i++)
            t.len[t.usable[i]] = ulens[i];
    }

    // Run-length code the lengths into (runcode, extra value, extra bits).
    struct RunOp { uint32_t code, extra; int extra_bits; };
    std::vector<RunOp> ops;
    for (uint32_t i = 0; i < sbnumsyms; ) {
        int v = t.len[i];
        uint32_t r = 1;
        while (i + r < sbnumsyms && t.len[i + r] == v)
            r++;
        if (v == 0) {
            while (r > 0) {
                if (r >= 11) {
                    uint32_t take = r > 138 ? 138 : r;
                    ops.push_back({ 34, take - 11, 7 });
                    r -= take; i += take;
                } else if (r >= 3) {
                    uint32_t take = r > 10 ? 10 : r;
                    ops.push_back({ 33, take - 3, 3 });
                    r -= take; i += take;
                } else {
                    ops.push_back({ 0, 0, 0 });
                    r--; i++;
                }
            }
        } else {
            // The first of a run is always spelled out; only then does
            // RUNCODE32's "copy the previous length" have a previous to copy.
            ops.push_back({ (uint32_t)v, 0, 0 });
            r--; i++;
            while (r > 0) {
                if (r >= 3) {
                    uint32_t take = r > 6 ? 6 : r;
                    ops.push_back({ 32, take - 3, 2 });
                    r -= take; i += take;
                } else {
                    ops.push_back({ (uint32_t)v, 0, 0 });
                    r--; i++;
                }
            }
        }
    }

    // Huffman the RUNCODE alphabet over what those ops actually used.
    std::vector<uint32_t> runcounts(35, 0);
    for (const RunOp &op : ops)
        runcounts[op.code]++;
    std::vector<uint32_t> rweights;
    std::vector<uint32_t> rindex;
    for (uint32_t i = 0; i < 35; i++)
        if (runcounts[i]) {
            rindex.push_back(i);
            rweights.push_back(runcounts[i]);
        }
    std::vector<int> runlens(35, 0);
    std::vector<int> rl = huffman_code_lengths(rweights, 15);
    for (size_t i = 0; i < rindex.size(); i++)
        runlens[rindex[i]] = rl[i];
    std::vector<uint32_t> runcodes;
    assign_canonical_codes(runlens, runcodes);

    BitWriter bw;
    for (int i = 0; i < 35; i++)
        bw_put_bits(bw, (uint32_t)runlens[(size_t)i], 4);   // step 1
    for (const RunOp &op : ops) {
        bw_put_bits(bw, runcodes[op.code], runlens[op.code]);
        if (op.extra_bits)
            bw_put_bits(bw, op.extra, op.extra_bits);
    }
    bw_finish(bw);   // step 6: instance coding resumes on a byte boundary
    append(d, bw.bytes.data(), bw.bytes.size());

    assign_canonical_codes(t.len, t.code);   // step 7
    return t;
}

// Composites a wi x hi source bitmap onto a dw x dh destination at (dx, dy),
// clipped to the destination bounds, using JBIG2's five combination
// operators (0 OR, 1 AND, 2 XOR, 3 XNOR, 4 REPLACE -- 7.4.1.5's encoding,
// matching JBig2ComposeOp). Mirrors the *observable* pixel-level behaviour
// of pdfium's CJBig2_Image::ComposeToInternal (jbig2_image.cpp), which
// implements the same clip-then-combine semantics through 32-bit-word
// bit-twiddling for speed; the per-pixel result is identical, only the
// implementation strategy differs.
static void compose_bitmap(std::vector<uint8_t> &dst, uint32_t dw, uint32_t dh,
                            const uint8_t *src, uint32_t sw, uint32_t sh,
                            int32_t dx, int32_t dy, uint8_t op)
{
    for (uint32_t sy = 0; sy < sh; sy++) {
        int32_t ty = dy + (int32_t)sy;
        if (ty < 0 || ty >= (int32_t)dh)
            continue;
        for (uint32_t sx = 0; sx < sw; sx++) {
            int32_t tx = dx + (int32_t)sx;
            if (tx < 0 || tx >= (int32_t)dw)
                continue;
            uint8_t s = src[(size_t)sy * sw + sx] ? 1 : 0;
            uint8_t &d = dst[(size_t)ty * dw + tx];
            switch (op) {
            case 0: d = (uint8_t)(d | s); break;
            case 1: d = (uint8_t)(d & s); break;
            case 2: d = (uint8_t)(d ^ s); break;
            case 3: d = (uint8_t)(1 - (d ^ s)); break;   // XNOR
            default: d = s; break;                       // REPLACE
            }
        }
    }
}

// 7.4.3.1 + 6.4: a real, decodable Huffman-coded (SBHUFF=1) text region,
// drawing from whichever real symbol dictionaries (SDHUFF=1/SDREFAGG=0
// ones from gen_symbol_dict_real) are available in `prior`, used by
// gen_segment_text_region() whenever that's the combination it randomly
// picks. One strip, SBSTRIPS=1 (so no bits are spent on the T coordinate
// per 6.4.9), holds every instance; each instance's S coordinate only ever
// moves forward, so the DS/FS deltas never need negative-range table
// support.
//
// `sbrefine` selects SBREFINE: with it set, each instance's RI bit (6.4.11)
// is coded, and a true RI is given a genuinely refined bitmap -- the
// referenced symbol's own real pixels (from `prior`'s ExportedSymbol data),
// refined via the MQ coder (mq_encode_refinement(), the same one the
// standalone refinement region uses) against a synthetic "current" bitmap.
// RDWI/RDHI/RDXI/RDYI are always coded as 0: Table 12's GRREFERENCEDX/DY
// only cancel out to a divisor-independent 0 in that case, sidestepping a
// real inconsistency between PDFium's Huffman-path and arithmetic-path
// CheckTRDReferenceDimension() shift amounts (2 vs 1, where the spec's own
// worked example confirms 1) that would otherwise misplace the reference
// for any nonzero delta.
SegResult gen_text_region_real(const std::vector<GeneratedSegment> &prior, bool sbrefine)
{
    // Real content needs the declared region size to agree with what's
    // actually placed (mirroring gen_segment_generic_region()'s and
    // gen_segment_refinement_region()'s small caps for their real-content
    // paths) -- the default 0x10000 cap lets width*height exceed what
    // CJBig2_Image can allocate (roughly INT_MAX pixels total), which makes
    // the very first allocation in DecodeHuffman fail before any of the
    // Huffman-coded content below is even read. 128 comfortably covers the
    // worst case here: up to 4 instances, each contributing at most an
    // 8-pixel S delta plus a symbol as wide as three 8-pixel width deltas.
    RegionInfo ri = gen_segment_region_info(false, 128);
    std::vector<uint8_t> d = ri.bytes;

    // Only dictionaries that actually exported symbols are candidates --
    // the structural (SDHUFF=0 / SDREFAGG=1) ones carry random payload, so
    // their symbols aren't real and can't be placed. SBNUMSYMS then follows
    // from whichever the strategy picked, rather than the selection being
    // fixed at "the first three".
    std::vector<uint32_t> dict_pool;
    for (const auto &seg : prior)
        if (seg.type == SEG_SYMBOL_DICTIONARY && seg.num_symbols > 0)
            dict_pool.push_back(seg.number);
    std::vector<uint32_t> refs = pick_refs(dict_pool, 1, 3);

    uint32_t sbnumsyms = 0;
    std::vector<ExportedSymbol> all_symbols;   // SBSYMS order: refs order, then within-dict order
    for (uint32_t r : refs)
        for (const auto &seg : prior)
            if (seg.number == r) {
                sbnumsyms += seg.num_symbols;
                for (const auto &es : seg.symbols)
                    all_symbols.push_back(es);
                break;
            }

    // 7.4.3.1.6: SBHUFFFS may also be a "user-supplied" (selector 3) table,
    // taken from a referred-to tables segment -- real, decodable content
    // needs that table to actually be usable, which means finding one whose
    // harvested line (GeneratedSegment::table_rows) is present. Only FS
    // needs this: it is coded exactly once per region (6.4.7's single first
    // S coordinate), so a table with only one genuinely-known-encodable
    // value is still enough to use it for real, unlike DS/DT which are
    // coded once per instance/strip and would need many known values.
    const GeneratedSegment *custom_fs_seg = nullptr;
    for (const auto &seg : prior)
        if (seg.type == SEG_TABLES && !seg.table_rows.empty()) {
            custom_fs_seg = &seg;
            break;
        }
    bool fs_table_b7 = (urand() & 1) != 0;      // false=B.6, true=B.7
    uint8_t ds_table_sel = (uint8_t)(urand() % 3);   // 0=B.8, 1=B.9, 2=B.10
    uint8_t dt_table_sel = (uint8_t)(urand() % 3);   // 0=B.11, 1=B.12, 2=B.13
    // Only meaningful when sbrefine; harmless (never read by a decoder) 0
    // otherwise, same as SBRTEMPLATE and the RDW/RDH/RDX/RDY table
    // selectors below.
    uint8_t sbrtemplate = sbrefine ? (uint8_t)(urand() % 2) : 0;

    // 7.4.3.1.1: text region segment flags. SBHUFF=1 (bit 0), SBREFINE
    // (bit 1), LOGSBSTRIPS=0 i.e. SBSTRIPS=1 (bits 2-3), REFCORNER random
    // (bits 4-5), TRANSPOSED=0 (bit 6), SBCOMBOP random (bits 7-8),
    // SBDEFPIXEL random (bit 9), SBDSOFFSET=0 (bits 10-14, kept simple),
    // SBRTEMPLATE (bit 15). refcorner/sbcombop/sbdefpixel are kept as named
    // values (not just folded into `flags`) because the composited region
    // bitmap built below needs them too.
    uint8_t refcorner = (uint8_t)(urand() % 4);   // JBig2Corner: 0 BL, 1 TL, 2 BR, 3 TR
    uint8_t sbcombop = (uint8_t)(urand() % 4);
    bool sbdefpixel = (urand() & 1) != 0;
    uint16_t flags = 0x0001;
    if (sbrefine)
        flags |= 0x0002;
    flags |= (uint16_t)(refcorner << 4);
    flags |= (uint16_t)(sbcombop << 7);
    if (sbdefpixel)
        flags |= 0x0200;
    flags |= (uint16_t)((uint32_t)sbrtemplate << 15);
    put_be16(d, flags);

    // 7.4.3.1.2: text region Huffman flags. SBHUFFFS selects a custom table
    // (selector 3) when one is available, else a standard one like DS/DT
    // always do; SBHUFFRDW/RDH/RDX/RDY/RSIZE stay 0 (Table
    // B.14/B.14/B.15/B.15/B.1) -- the values gen_text_region_real() always
    // codes below, real refined instances included.
    uint16_t hflags = 0;
    hflags |= (uint16_t)((custom_fs_seg ? 3 : (fs_table_b7 ? 1 : 0)) << 0);
    hflags |= (uint16_t)(ds_table_sel << 2);
    hflags |= (uint16_t)(dt_table_sel << 4);
    put_be16(d, hflags);
    if (custom_fs_seg)
        refs.push_back(custom_fs_seg->number);

    // 7.4.3.1.3: refinement AT flags, present only if SBREFINE and
    // SBRTEMPLATE == 0. Shared by every refined instance in this region
    // (Table 12: GRATXn/GRATYn = SBRATXn/SBRATYn), unlike RDW/RDH/RDX/RDY
    // which are per-instance.
    AtPixel at1 = {}, at2 = {};
    if (sbrefine && sbrtemplate == 0) {
        at1 = write_primary_at_pixel(d);       // SBRATX1/SBRATY1
        at2 = write_reference_at_pixel(d);     // SBRATX2/SBRATY2
    }

    uint32_t sbnuminstances = sbnumsyms > 0 ? 1 + (urand() % 4) : 0;
    put_be32(d, sbnuminstances);   // 7.4.3.1.4: SBNUMINSTANCES

    SymbolIDTable symtab = write_symbol_id_table(d, sbnumsyms);   // 7.4.3.1.5/.1.7

    // A real decoder passes one GRCONTEXTS span to every refined instance's
    // generic refinement region decode in this text region (jbig2_trd_proc's
    // DecodeHuffman(), matching how a symbol dictionary's SDD proc shares
    // its own grContexts across new symbols): the adaptive probability
    // state accumulates across instances, even though each instance's raw
    // arithmetic coder itself restarts fresh at a byte boundary. Two
    // separate persisting arrays here -- one for encoding, one for the
    // parallel "shadow decode" mq_finalize_refinement() runs to measure
    // RSIZE -- mirror that single shared array; using a fresh array per
    // instance instead (as a naive per-call allocation would) desyncs the
    // second and later refined instances from what a real decode does.
    std::vector<uint8_t> refine_cx_state, refine_measure_cx_state;
    if (sbrefine) {
        size_t sz = 1u << (sbrtemplate == 0 ? 13 : 10);
        refine_cx_state.assign(sz, 0);
        refine_measure_cx_state.assign(sz, 0);
    }

    // A custom FS table has exactly one line (the harvested one) -- correct
    // to encode with, since the only value ever coded through it is that
    // line's own `val` (below), which huff_encode() then finds with a
    // zero-bit offset regardless of the line's actual range_bits.
    StdHuffTable fs_table = custom_fs_seg ? StdHuffTable{ custom_fs_seg->table_rows.data(), 1 }
                           : fs_table_b7 ? STD_TABLE(HUFF_B7) : STD_TABLE(HUFF_B6);
    StdHuffTable ds_table = ds_table_sel == 2 ? STD_TABLE(HUFF_B10)
                           : ds_table_sel == 1 ? STD_TABLE(HUFF_B9) : STD_TABLE(HUFF_B8);
    StdHuffTable dt_table = dt_table_sel == 2 ? STD_TABLE(HUFF_B13)
                           : dt_table_sel == 1 ? STD_TABLE(HUFF_B12) : STD_TABLE(HUFF_B11);

    // 6.4.5 step 2: the initial STRIPT value is delta-T coded (6.4.6) and
    // then *negated*, so the 1 below starts STRIPT at -1; the first
    // strip's own delta T of 1 (step 4b) brings it back to 0, putting that
    // strip at T = 0. Neither can simply be coded as 0: B.11-B.13, the
    // only SBHUFFDT choices (Table 33), all start at 1, so 0 has no
    // representation -- the spec's own worked example likewise codes an
    // initial value of 1 to reach a first strip at STRIPT + delta.
    // 6.4.5-8's placement math, replicated here so the composited region
    // bitmap below (r.bitmap) ends up holding exactly what a decoder's own
    // ComposeTo() calls would produce -- see GetComposeData()
    // (jbig2_trd_proc.cpp) for the TRANSPOSED==0 formulas this mirrors:
    // FIRSTS/CURS track the same running S position a decoder maintains,
    // and a right-aligned REFCORNER shifts CURS by WI-1 *before* deriving
    // the placement X, while a left-aligned one shifts it *after* (via
    // `increment`) -- getting either half backwards silently staggers
    // every instance after the first.
    bool right_corner = (refcorner == 2 || refcorner == 3);   // BOTTOMRIGHT, TOPRIGHT
    bool bottom_corner = (refcorner == 0 || refcorner == 2);  // BOTTOMLEFT, BOTTOMRIGHT
    int32_t firsts = 0, curs = 0;
    std::vector<uint8_t> canvas((size_t)ri.width * ri.height, sbdefpixel ? 1 : 0);

    uint32_t nrefined = 0;
    BitWriter bw;
    huff_encode(bw, dt_table, 1);
    if (sbnuminstances > 0) {
        huff_encode(bw, dt_table, 1);   // this strip's delta T
        for (uint32_t i = 0; i < sbnuminstances; i++) {
            if (i == 0) {
                // With a custom table the only representable value is its
                // harvested line's own `val` -- anything else would fail
                // huff_encode()'s representability check.
                int32_t dfs = custom_fs_seg ? custom_fs_seg->table_rows[0].val
                                             : 1 + (int32_t)(urand() % 8);
                huff_encode(bw, fs_table, dfs);   // 6.4.7: first S coordinate
                firsts += dfs;
                curs = firsts;
            } else {
                int32_t ids = (int32_t)(urand() % 8);
                huff_encode(bw, ds_table, ids);   // 6.4.8: subsequent S coordinate
                curs += ids;   // SBDSOFFSET == 0
            }
            // 6.4.9: T coordinate -- no bits consumed, SBSTRIPS == 1, and
            // STRIPT is fixed at 0 (see the comment on the initial delta-T
            // below), so TI is always 0 too.
            // 6.4.10: symbol ID, as its SBSYMCODES code. Only a symbol the
            // table gave a code to can be named -- a zero-length entry has
            // none, so it stays out of the draw.
            uint32_t idi = symtab.usable[urand() % symtab.usable.size()];
            bw_put_bits(bw, symtab.code[idi], symtab.len[idi]);

            const ExportedSymbol &sym = all_symbols[idi];
            const uint8_t *place_px = sym.px.data();   // IBI = SBSYMS[IDI] when RI == 0

            // 6.4.11: symbol instance bitmap. RI is only coded at all when
            // SBREFINE is set; a false RI (or SBREFINE off) leaves IBI as
            // SBSYMS[IDI] unmodified, needing nothing further here.
            std::vector<uint8_t> refined_cur;   // keeps `place_px` valid past this scope
            if (sbrefine) {
                bool ri = (urand() & 1) != 0;
                bw_put_bits(bw, ri ? 1 : 0, 1);
                if (ri) {
                    nrefined++;
                    // RDWI/RDHI/RDXI/RDYI = 0: WOI/HOI equal the reference
                    // symbol's own size (Table 12's CheckTRDDimension), and
                    // GRREFERENCEDX/DY collapse to 0 regardless of the
                    // divisor -- see the function comment. A coarse-block
                    // "current" bitmap (not a copy of the reference) gives
                    // the refinement coder genuine new content to code
                    // against a real reference.
                    refined_cur.resize((size_t)sym.w * sym.h);
                    for (uint32_t y = 0; y < sym.h; y++)
                        for (uint32_t x = 0; x < sym.w; x++)
                            refined_cur[(size_t)y * sym.w + x] = (uint8_t)(((x >> 1) + (y >> 1)) & 1);
                    std::vector<uint8_t> coded = mq_encode_refinement(
                        (int)sym.w, (int)sym.h, refined_cur.data(), sym.px.data(),
                        sbrtemplate, /*tpgron=*/false, at1.x, at1.y, at2.x, at2.y,
                        &refine_cx_state);
                    // RSIZE must equal the exact number of bytes a real
                    // decode consumes, which BYTEIN's 0xFF-marker handling
                    // makes content-dependent -- not simply coded.size()
                    // (confirmed empirically: the two can differ by several
                    // bytes either way). mq_finalize_refinement() measures
                    // it by actually decoding `coded` back.
                    uint32_t rsize;
                    std::vector<uint8_t> final_bytes = mq_finalize_refinement(
                        (int)sym.w, (int)sym.h, sym.px.data(), sbrtemplate,
                        at1.x, at1.y, at2.x, at2.y, coded, &rsize, &refine_measure_cx_state);

                    huff_encode(bw, STD_TABLE(HUFF_B14), 0);   // RDWI
                    huff_encode(bw, STD_TABLE(HUFF_B14), 0);   // RDHI
                    huff_encode(bw, STD_TABLE(HUFF_B15), 0);   // RDXI
                    huff_encode(bw, STD_TABLE(HUFF_B15), 0);   // RDYI
                    // 6.4.11.5/step 5b: RSIZE, then byte-align -- the
                    // refinement bitmap's arithmetic coding is byte-based,
                    // so it (and the 2 bytes below) must start on a byte
                    // boundary the same way the collective bitmaps in
                    // gen_symbol_dict_real() do.
                    huff_encode(bw, STD_TABLE(HUFF_B1), (int32_t)rsize);
                    bw_finish(bw);
                    append(d, bw.bytes.data(), bw.bytes.size());
                    bw = BitWriter();

                    append(d, final_bytes.data(), final_bytes.size());
                    // 2 trailing bytes a decoder unconditionally skips over
                    // (matching the +2 already folded into `rsize`) --
                    // *without* inspecting their value for that skip, but
                    // their value still matters one step earlier: if
                    // `final_bytes` itself ends in 0xFF, BYTEIN's
                    // marker-detection peeks at exactly this next byte to
                    // decide whether to freeze its byte position, so this
                    // must be the same 0xFF mq_finalize_refinement() assumed
                    // was here when it measured `final_bytes`'s length --
                    // anything else (e.g. 0x00) can make the real decode
                    // consume a different number of bytes than measured.
                    d.push_back(0xFF);
                    d.push_back(0xFF);

                    // mq_encode_refinement() would rewrite refined_cur where
                    // TPGRON skipped a pixel, but tpgron is always false
                    // here, so refined_cur already holds exactly what a
                    // decoder reconstructs -- verified in the same way as
                    // the standalone refinement region (round-tripped
                    // through pdfium's real decoder).
                    place_px = refined_cur.data();
                }
            }

            if (right_corner)
                curs += (int32_t)sym.w - 1;
            int32_t si = curs;
            int32_t dst_x = right_corner ? si - (int32_t)sym.w + 1 : si;
            int32_t dst_y = bottom_corner ? -(int32_t)sym.h + 1 : 0;   // TI == 0
            compose_bitmap(canvas, ri.width, ri.height, place_px, sym.w, sym.h,
                           dst_x, dst_y, sbcombop);
            if (!right_corner)
                curs += (int32_t)sym.w - 1;
        }
        huff_encode(bw, ds_table, OOB_VAL);   // end of strip
    }
    bw_finish(bw);
    append(d, bw.bytes.data(), bw.bytes.size());

    printf("text-region handler (%zu bytes, %u instances, %u refined, %u available symbols, %zu refs, real content%s)\n",
           d.size(), sbnuminstances, nrefined, sbnumsyms, refs.size(),
           custom_fs_seg ? ", custom FS table" : "");
    SegResult r;
    r.data = d;
    r.refs = refs;
    r.colored = ri.colored;
    r.region_x = ri.x;
    r.region_y = ri.y;
    r.combop = ri.combop;
    // The full composited region bitmap, exactly as a decoder's ComposeTo()
    // calls would leave it -- only useful to a later segment when *this*
    // one ends up being the intermediate variant (7.4.3.1/Figure 37 is
    // shared verbatim across intermediate/immediate/immediate-lossless), so
    // it's built unconditionally here and left to the caller (a refinement
    // region's own pool-building code, checking the resolved segment type)
    // to decide whether it's actually reachable -- the same pattern
    // gen_segment_generic_region() already uses for its own r.bitmap.
    r.bw = ri.width;
    r.bh = ri.height;
    r.bitmap = std::move(canvas);
    return r;
}

// 7.4.3.1 + 6.4: a real, decodable arithmetic-coded (SBHUFF=0) text
// region. Same shape as gen_text_region_real() -- SBSTRIPS=1 (one strip
// holds every instance), RDWI=RDHI=RDXI=RDYI=0 for refined instances (Table
// 12's CheckTRDDimension/CheckTRDReferenceDimension collapse those to the
// reference symbol's own size at zero offset regardless of the shift
// amount) -- but every structural integer (the initial/per-strip delta T,
// FS, DS, RI, and the per-instance symbol ID) goes through Annex A.2/A.3's
// procedures (IADT/IAFS/IADS/IARI/IAID) instead of a Huffman table, and a
// refined instance's own refinement coding is embedded directly in the
// same one continuous arithmetic stream (mq_encode_refinement_into(), no
// RSIZE field or byte-alignment reset needed -- jbig2_trd_proc.cpp's
// arithmetic DecodeArith passes its *own* pArithDecoder straight through
// to the refinement decode, unlike SBHUFF=1's per-instance byte-aligned
// sub-stream). IADT/IAFS/IADS/IARI/IARDW/IARDH/IARDX/IARDY are each one
// persistent context across the *whole* region (JBig2IntDecoderState is
// constructed once and its pointers extracted once, before the per
// -instance loop, in both DecodeArith and here) -- unlike the per-instance
// -fresh mq_encode_refinement() context reset a naive per-instance
// declaration would give.
SegResult gen_text_region_real_arith(const std::vector<GeneratedSegment> &prior, bool sbrefine)
{
    RegionInfo ri = gen_segment_region_info(false, 128);
    std::vector<uint8_t> d = ri.bytes;

    std::vector<uint32_t> dict_pool;
    for (const auto &seg : prior)
        if (seg.type == SEG_SYMBOL_DICTIONARY && seg.num_symbols > 0)
            dict_pool.push_back(seg.number);
    std::vector<uint32_t> refs = pick_refs(dict_pool, 1, 3);

    uint32_t sbnumsyms = 0;
    std::vector<ExportedSymbol> all_symbols;
    for (uint32_t r : refs)
        for (const auto &seg : prior)
            if (seg.number == r) {
                sbnumsyms += seg.num_symbols;
                for (const auto &es : seg.symbols)
                    all_symbols.push_back(es);
                break;
            }

    uint8_t sbrtemplate = sbrefine ? (uint8_t)(urand() % 2) : 0;

    // 7.4.3.1.1: text region segment flags. SBHUFF=0 (bit 0), SBREFINE
    // (bit 1), LOGSBSTRIPS=0 i.e. SBSTRIPS=1 (bits 2-3), REFCORNER random
    // (bits 4-5), TRANSPOSED=0 (bit 6), SBCOMBOP random (bits 7-8),
    // SBDEFPIXEL random (bit 9), SBDSOFFSET=0 (bits 10-14), SBRTEMPLATE
    // (bit 15).
    uint8_t refcorner = (uint8_t)(urand() % 4);
    uint8_t sbcombop = (uint8_t)(urand() % 4);
    bool sbdefpixel = (urand() & 1) != 0;
    uint16_t flags = 0x0000;
    if (sbrefine)
        flags |= 0x0002;
    flags |= (uint16_t)(refcorner << 4);
    flags |= (uint16_t)(sbcombop << 7);
    if (sbdefpixel)
        flags |= 0x0200;
    flags |= (uint16_t)((uint32_t)sbrtemplate << 15);
    put_be16(d, flags);

    // 7.4.3.1.2: text region Huffman flags field -- absent, SBHUFF == 0.

    // 7.4.3.1.3: refinement AT flags, present only if SBREFINE and
    // SBRTEMPLATE == 0.
    AtPixel at1 = {}, at2 = {};
    if (sbrefine && sbrtemplate == 0) {
        at1 = write_primary_at_pixel(d);       // SBRATX1/SBRATY1
        at2 = write_reference_at_pixel(d);     // SBRATX2/SBRATY2
    }

    uint32_t sbnuminstances = sbnumsyms > 0 ? 1 + (urand() % 4) : 0;
    put_be32(d, sbnuminstances);   // 7.4.3.1.4: SBNUMINSTANCES

    // SBSYMCODELEN starts at 0 for an arithmetic text region (unlike a
    // symbol dictionary's own SBSYMCODELEN, which starts at 1) --
    // jbig2_context.cpp:722-726 -- so a single-symbol region (SBNUMSYMS ==
    // 1) needs 0 IAID bits, always decoding IDI == 0.
    int sbsymcodelen = 0;
    while ((1u << sbsymcodelen) < sbnumsyms)
        sbsymcodelen++;

    // One continuous arithmetic-coded stream: every named procedure below
    // is one persistent context across the whole region, matching
    // JBig2IntDecoderState's single set of decoders (jbig2_trd_proc.h).
    MQEncoder mq;
    ArithIntCtx iadt, iafs, iads, iari, iardw, iardh, iardx, iardy;
    std::vector<uint8_t> iaid_cx(1u << sbsymcodelen, 0);
    std::vector<uint8_t> refine_cx_state;
    if (sbrefine)
        refine_cx_state.assign(1u << (sbrtemplate == 0 ? 13 : 10), 0);

    // 6.4.5-8's placement math -- see gen_text_region_real()'s comment on
    // the identical FIRSTS/CURS/GetComposeData logic this mirrors.
    bool right_corner = (refcorner == 2 || refcorner == 3);
    bool bottom_corner = (refcorner == 0 || refcorner == 2);
    int32_t firsts = 0, curs = 0;
    std::vector<uint8_t> canvas((size_t)ri.width * ri.height, sbdefpixel ? 1 : 0);

    uint32_t nrefined = 0;
    // 6.4.5 step 2/4b: INITIAL STRIPT and this (one) strip's own delta T
    // are both coded as plain 0 -- arithmetic integer coding has no
    // B.11-B.13-style "can't represent 0" gap the way those standard
    // Huffman tables do (gen_text_region_real()'s comment), so unlike
    // there, no "start at -1, offset back to 0" trick is needed.
    mq_encode_arith_int(mq, iadt, 0);
    if (sbnuminstances > 0) {
        mq_encode_arith_int(mq, iadt, 0);   // this strip's delta T
        for (uint32_t i = 0; i < sbnuminstances; i++) {
            if (i == 0) {
                int32_t dfs = 1 + (int32_t)(urand() % 8);
                mq_encode_arith_int(mq, iafs, dfs);   // 6.4.7: first S coordinate
                firsts += dfs;
                curs = firsts;
            } else {
                int32_t ids = (int32_t)(urand() % 8);
                mq_encode_arith_int(mq, iads, ids);   // 6.4.8: subsequent S coordinate
                curs += ids;   // SBDSOFFSET == 0
            }
            // 6.4.9: T coordinate -- SBSTRIPS == 1 skips IAIT entirely
            // (jbig2_trd_proc.cpp:322), so TI is always STRIPT (== 0) too.
            uint32_t idi = urand() % sbnumsyms;
            mq_encode_arith_iaid(mq, iaid_cx, sbsymcodelen, idi);   // 6.4.10

            const ExportedSymbol &sym = all_symbols[idi];
            const uint8_t *place_px = sym.px.data();   // IBI = SBSYMS[IDI] when RI == 0

            // 6.4.11: symbol instance bitmap. RI is only coded at all when
            // SBREFINE is set (else it's implicitly 0, no IARI call --
            // jbig2_trd_proc.cpp:339-343).
            std::vector<uint8_t> refined_cur;
            if (sbrefine) {
                // GRTEMPLATE == 1 refinement always hits pdfium's
                // DecodeTemplate1Opt here (GRREFERENCEDX == 0 and GRW ==
                // reference width always hold, since RDW/RDX are always 0
                // below). That routine reads the reference bitmap a whole
                // byte at a time and does NOT mask off the bits beyond the
                // symbol's real width in a row's last byte -- it trusts
                // that padding is zero. Symbols exported from a Huffman
                // symbol dictionary can carry non-zero bits there (an
                // artifact of how collective bitmaps are split), which we
                // have no way to predict/replicate at encode time. Avoid
                // the whole class of divergence by only refining symbols
                // whose width is byte-aligned, so no row ever has a
                // partial last byte for Opt to misread.
                bool width_safe_for_refine = sbrtemplate != 1 || (sym.w % 8) == 0;
                bool ri_bit = width_safe_for_refine && (urand() & 1) != 0;
                mq_encode_arith_int(mq, iari, ri_bit ? 1 : 0);
                if (ri_bit) {
                    nrefined++;
                    // RDWI/RDHI/RDXI/RDYI = 0 (see the function comment); a
                    // coarse-block "current" bitmap gives the refinement
                    // coder genuine new content to code against a real
                    // reference, same as gen_text_region_real()'s SBHUFF=1
                    // refined instances.
                    refined_cur.resize((size_t)sym.w * sym.h);
                    for (uint32_t y = 0; y < sym.h; y++)
                        for (uint32_t x = 0; x < sym.w; x++)
                            refined_cur[(size_t)y * sym.w + x] = (uint8_t)(((x >> 1) + (y >> 1)) & 1);
                    mq_encode_arith_int(mq, iardw, 0);
                    mq_encode_arith_int(mq, iardh, 0);
                    mq_encode_arith_int(mq, iardx, 0);
                    mq_encode_arith_int(mq, iardy, 0);
                    mq_encode_refinement_into(mq, refine_cx_state, (int)sym.w, (int)sym.h,
                                               refined_cur.data(), sym.px.data(),
                                               sbrtemplate, /*tpgron=*/false,
                                               at1.x, at1.y, at2.x, at2.y);
                    place_px = refined_cur.data();
                }
            }

            if (right_corner)
                curs += (int32_t)sym.w - 1;
            int32_t si = curs;
            int32_t dst_x = right_corner ? si - (int32_t)sym.w + 1 : si;
            int32_t dst_y = bottom_corner ? -(int32_t)sym.h + 1 : 0;   // TI == 0
            compose_bitmap(canvas, ri.width, ri.height, place_px, sym.w, sym.h,
                           dst_x, dst_y, sbcombop);
            if (!right_corner)
                curs += (int32_t)sym.w - 1;
        }
        mq_encode_arith_int(mq, iads, 0, /*oob=*/true);   // end of strip
    }
    mq_flush(mq);
    append(d, mq.out.data(), mq.out.size());
    d.push_back(0xFF);
    d.push_back(0xFF);

    printf("text-region handler (%zu bytes, %u instances, %u refined, %u available symbols, %zu refs, real arithmetic content)\n",
           d.size(), sbnuminstances, nrefined, sbnumsyms, refs.size());
    SegResult r;
    r.data = d;
    r.refs = refs;
    r.colored = ri.colored;
    r.region_x = ri.x;
    r.region_y = ri.y;
    r.combop = ri.combop;
    r.bw = ri.width;
    r.bh = ri.height;
    r.bitmap = std::move(canvas);
    return r;
}

// 7.4.3.1: text region segment data header (Figure 37). SBHUFF and
// SBREFINE are picked from the full range the spec allows; both SBREFINE
// combinations under both SBHUFF values are real, decodable content when
// `prior` has real symbols to place -- gen_text_region_real() (SBHUFF=1)
// and gen_text_region_real_arith() (SBHUFF=0) -- see their own comments
// for how refined instances work. Without real symbols available, both
// fall back to the structural path.
SegResult gen_segment_text_region(const std::vector<GeneratedSegment> &prior)
{
    bool sbhuff = (urand() & 1) != 0;
    bool sbrefine = (urand() & 1) != 0;

    // gen_text_region_real() can only build a decodable region if some
    // earlier symbol dictionary actually exported symbols for it to place.
    // With none, it emits SBNUMINSTANCES = 0 and a symbol ID table that is
    // just its fixed 35-entry runcode header, which decoders reject
    // (pdfium's DecodeSymbolIDHuffmanTable returns empty, failing
    // ParseTextRegion) -- a Huffman-specific parsing quirk gen_text_region_
    // real_arith() doesn't actually share (IAID's context array is valid
    // even when SBSYMCODELEN == 0, so an arithmetic region with no symbols
    // decodes fine, just empty), but the same gate is used for it anyway
    // to prefer real, non-degenerate content over trivially-valid-but-empty
    // content when symbols *are* available. Since main() picks segment
    // types independently, a text region drawn before any real symbol
    // dictionary would otherwise usually have nothing to place.
    bool have_real_syms = false;
    for (const auto &seg : prior)
        if (seg.type == SEG_SYMBOL_DICTIONARY && seg.num_symbols > 0) {
            have_real_syms = true;
            break;
        }

    if (have_real_syms)
        return sbhuff ? gen_text_region_real(prior, sbrefine)
                      : gen_text_region_real_arith(prior, sbrefine);

    RegionInfo ri = gen_segment_region_info();
    std::vector<uint8_t> d = ri.bytes;

    // 7.4.3.2: a text region draws its symbol instances from the symbol
    // dictionaries it refers to; a real one needs at least one.
    std::vector<uint32_t> dict_pool = segment_numbers_of_type(prior, SEG_SYMBOL_DICTIONARY);
    std::vector<uint32_t> refs = pick_refs(dict_pool, 1, 3);   // a text region needs symbols

    uint8_t sbrtemplate = sbrefine ? (uint8_t)(urand() % 2) : 0;

    // 7.4.3.1.1: text region segment flags (2 bytes).
    uint16_t flags = 0;
    if (sbhuff)
        flags |= 0x0001;
    if (sbrefine)
        flags |= 0x0002;
    flags |= (uint16_t)((urand() % 4) << 2);    // LOGSBSTRIPS
    flags |= (uint16_t)((urand() % 4) << 4);    // REFCORNER
    if (urand() & 1)
        flags |= 0x0040;                        // TRANSPOSED
    flags |= (uint16_t)((urand() % 4) << 7);    // SBCOMBOP
    if (urand() & 1)
        flags |= 0x0200;                        // SBDEFPIXEL
    flags |= (uint16_t)((urand() & 0x1F) << 10); // SBDSOFFSET (signed 5-bit)
    flags |= (uint16_t)(sbrtemplate << 15);
    put_be16(d, flags);

    // 7.4.3.1.6: "user-supplied" (3) Huffman selectors must each be matched
    // by a distinct referred-to tables segment, in the order SBHUFFFS,
    // SBHUFFDS, SBHUFFDT, SBHUFFRDW, SBHUFFRDH, SBHUFFRDX, SBHUFFRDY,
    // SBHUFFRSIZE.
    std::vector<uint32_t> table_pool = segment_numbers_of_type(prior, SEG_TABLES);
    std::vector<uint32_t> table_refs;

    // 7.4.3.1.2: text region Huffman flags (2 bytes), present only if SBHUFF.
    if (sbhuff) {
        uint16_t hflags = 0;
        hflags |= (uint16_t)(pick_sel_013(table_pool, table_refs) << 0);                     // SBHUFFFS
        hflags |= (uint16_t)(pick_sel_0123(table_pool, table_refs) << 2);                    // SBHUFFDS
        hflags |= (uint16_t)(pick_sel_0123(table_pool, table_refs) << 4);                    // SBHUFFDT
        hflags |= (uint16_t)((sbrefine ? pick_sel_013(table_pool, table_refs) : 0) << 6);     // SBHUFFRDW
        hflags |= (uint16_t)((sbrefine ? pick_sel_013(table_pool, table_refs) : 0) << 8);     // SBHUFFRDH
        hflags |= (uint16_t)((sbrefine ? pick_sel_013(table_pool, table_refs) : 0) << 10);    // SBHUFFRDX
        hflags |= (uint16_t)((sbrefine ? pick_sel_013(table_pool, table_refs) : 0) << 12);    // SBHUFFRDY
        if (sbrefine && pick_sel_bit(table_pool, table_refs))
            hflags |= 0x4000;                                             // SBHUFFRSIZE
        put_be16(d, hflags);
    }

    // 7.4.3.1.3: text region refinement AT flags, present only if
    // SBREFINE == 1 and SBRTEMPLATE == 0.
    if (sbrefine && sbrtemplate == 0) {
        write_primary_at_pixel(d);       // SBRATX1/SBRATY1
        write_reference_at_pixel(d);     // SBRATX2/SBRATY2
    }

    put_be32(d, urand() % 256);          // 7.4.3.1.4: SBNUMINSTANCES

    // 7.4.3.1.5/.1.7: symbol ID Huffman decoding table, present only if
    // SBHUFF; it is itself entropy-coded, so it's not modeled precisely.
    if (sbhuff)
        append_random_payload(d, 64);

    // Remainder: coded symbol instance data, not modeled.
    append_random_payload(d, 256);

    for (uint32_t t : table_refs)
        refs.push_back(t);

    printf("text-region handler (%zu bytes, %zu refs, structural)\n", d.size(), refs.size());
    SegResult r;
    r.data = d;
    r.refs = refs;
    r.colored = ri.colored;
    r.combop = ri.combop;
    r.region_x = ri.x;
    r.region_y = ri.y;
    return r;
}

// 7.4.5.1: halftone region segment data header (Figure 43). Starts with
// the common region segment information field (7.4.1).
// 7.4.5.1 + 6.6: a real, decodable halftone region, whenever `prior` has a
// real pattern dictionary (gen_segment_pattern_dict()'s content) to refer
// to. HGX=HGY=0 and HRX=256 (this format's 8.8 fixed-point for "1.0"),
// HRY=0 place grid cell (ng, mg) at exactly pixel (ng, mg): 6.6.5.1's
// x = (HGX + mg*HRY + ng*HRX) >> 8, y = (HGY + mg*HRX - ng*HRY) >> 8, both
// collapse to a plain grid index once HRY is 0 (jbig2_htrd_proc.cpp's
// DecodeImage/HSKIP loops use this exact formula).
SegResult gen_segment_halftone_region(const std::vector<GeneratedSegment> &prior)
{
    // 7.4.5.2: a halftone region decodes patterns from its referred-to
    // pattern dictionary segment; exactly one, so min and max are both 1 --
    // the strategies still decide *which* one (oldest, newest, or any).
    std::vector<uint32_t> pat_pool = segment_numbers_of_type(prior, SEG_PATTERN_DICTIONARY);
    std::vector<uint32_t> refs = pick_refs(pat_pool, 1, 1);

    const GeneratedSegment *patseg = nullptr;
    if (!refs.empty())
        for (const auto &seg : prior)
            if (seg.number == refs[0] && seg.hdpw > 0) {
                patseg = &seg;
                break;
            }

    bool real = patseg != nullptr;
    std::vector<uint8_t> d;
    // 7.4.5.1.1: halftone region segment flags (1 byte).
    bool hmmr = (urand() & 1) != 0;
    uint8_t htemplate = hmmr ? 0 : (uint8_t)(urand() % 4);
    // HENABLESKIP only has a defined bitstream effect for arithmetic
    // content (jbig2_htrd_proc.cpp's DecodeMMR never builds or consults a
    // skip bitmap at all -- only DecodeArith does), so it stays off
    // whenever HMMR is set: setting it there would change nothing
    // decodable, only add an untested combination.
    bool henableskip = real && !hmmr && (urand() & 1);
    uint8_t hcombop = (uint8_t)(urand() % 5);
    bool hdefpixel = (urand() & 1) != 0;
    uint8_t flags = hmmr ? 0x01 : 0x00;
    flags |= (uint8_t)(htemplate << 1);
    if (henableskip)
        flags |= 0x08;
    flags |= (uint8_t)(hcombop << 4);
    if (hdefpixel)
        flags |= 0x80;
    d.push_back(flags);

    // Grid at least 2x2 when real, so a skip test always has both an
    // in-bounds cell (ng=mg=0) and an out-of-bounds one to force below,
    // regardless of how big HGW/HGH end up.
    uint32_t hgw = real ? 2 + (urand() % 4) : (urand() % 0x10000);
    uint32_t hgh = real ? 2 + (urand() % 4) : (urand() % 0x10000);
    put_be32(d, hgw);                              // HGW
    put_be32(d, hgh);                               // HGH
    put_be32(d, real ? 0 : urand());                 // HGX
    put_be32(d, real ? 0 : urand());                 // HGY
    put_be16(d, (uint16_t)(real ? 256 : urand()));    // HRX
    put_be16(d, (uint16_t)(real ? 0 : urand()));      // HRY

    // 6.6.5.1's out-of-bounds test simplifies, under this generator's
    // unit grid (x=ng, y=mg, HPW/HPH > 0 always), to just ng >= HBW or
    // mg >= HBH -- so shrinking the declared region one cell short of the
    // grid's own extent forces exactly the last row and column of cells
    // out of bounds, giving HSKIP a genuine (neither all-0 nor all-1) mix
    // to test instead of leaving it vacuously all-0. force_w/force_h (not
    // the usual random draw) is what makes that shrink exact.
    RegionInfo ri = henableskip
        ? gen_segment_region_info(false, 0, hgw - 1, hgh - 1)
        : gen_segment_region_info(false, real ? 32 : 0x10000);
    // The region-info bytes come before the flags byte (7.4.5.1), so they
    // have to be spliced in ahead of everything already written to `d`.
    d.insert(d.begin(), ri.bytes.begin(), ri.bytes.end());

    SegResult r;
    if (real) {
        uint32_t hnumpats = patseg->bw / patseg->hdpw;
        uint32_t hpw = patseg->hdpw;
        uint32_t hph = patseg->bh;

        // 6.6.5.1 step 1: GSBPP, same formula as pdfium's own HBPP.
        uint32_t gsbpp = 1;
        while ((1u << gsbpp) < hnumpats)
            gsbpp++;

        std::vector<uint8_t> skipmask;
        if (henableskip) {
            skipmask.assign((size_t)hgw * hgh, 0);
            for (uint32_t mg = 0; mg < hgh; mg++)
                for (uint32_t ng = 0; ng < hgw; ng++)
                    if (ng >= ri.width || mg >= ri.height)
                        skipmask[(size_t)mg * hgw + ng] = 1;
        }

        // Per-cell target pattern index (arbitrary but deterministic, like
        // every other real-content generator's synthetic pixel choices),
        // Gray-coded into GSBPP bitplanes MSB (gsbpp-1) first: plane
        // gsbpp-1 carries the raw top bit, every plane below it carries
        // that bit XOR'd against the *next* plane's own (undecoded) bit --
        // exactly what CJBig2_HTRDProc::DecodeArith's
        // `GSPLANES[i]->ComposeFrom(0, 0, GSPLANES[i + 1].get(),
        // JBIG2_COMPOSE_XOR)` undoes on the way back to reconstruct gsval.
        // A skipped cell's planes all stay 0 (never assigned below): a
        // decoder never spends a bit on it in any plane, so it always
        // reconstructs gsval 0 there regardless of what this loop would
        // otherwise have chosen.
        std::vector<std::vector<uint8_t>> planes(gsbpp, std::vector<uint8_t>((size_t)hgw * hgh));
        for (uint32_t mg = 0; mg < hgh; mg++) {
            for (uint32_t ng = 0; ng < hgw; ng++) {
                if (henableskip && skipmask[(size_t)mg * hgw + ng])
                    continue;
                uint32_t gidx = (ng + mg) % hnumpats;
                uint8_t prev_bit = 0;
                for (int i = (int)gsbpp - 1; i >= 0; i--) {
                    uint8_t bit = (uint8_t)((gidx >> i) & 1);
                    planes[i][(size_t)mg * hgw + ng] = bit ^ prev_bit;
                    prev_bit = bit;
                }
            }
        }

        std::vector<uint8_t> payload;
        if (hmmr) {
            // Each bitplane is its own T.6-coded image; a decoder
            // unconditionally re-aligns and skips 3 bytes after every one
            // (jbig2_htrd_proc.cpp's DecodeMMR), matching the fixed 2-byte
            // skip the MQ-coded paths elsewhere in this generator need
            // after their own trailing marker -- this is T.6's EOFB
            // equivalent, skipped without being inspected.
            for (int i = (int)gsbpp - 1; i >= 0; i--) {
                std::vector<uint8_t> coded = mmr_encode((int)hgw, (int)hgh, planes[i].data());
                append(payload, coded.data(), coded.size());
                payload.push_back(0);
                payload.push_back(0);
                payload.push_back(0);
            }
        } else {
            // Unlike every other real generic-region content in this
            // generator, every bitplane here shares *one* continuous
            // arithmetic-coded stream and one adaptive context array --
            // DecodeArith (jbig2_htrd_proc.cpp) reuses the same
            // CJBig2_ArithDecoder and gbContexts span across the whole
            // GSBPP loop, with no per-plane reset -- hence
            // mq_encode_generic_template{0,123}_into() instead of the
            // usual one-shot wrappers, and a single mq_flush() at the end.
            int atx1 = htemplate <= 1 ? 3 : 2, aty1 = -1;
            MQEncoder mq;
            std::vector<uint8_t> cx_state(
                1u << (htemplate == 0 ? 16 : htemplate == 1 ? 13 : 10), 0);
            const uint8_t *skip = henableskip ? skipmask.data() : nullptr;
            for (int i = (int)gsbpp - 1; i >= 0; i--) {
                if (htemplate == 0)
                    mq_encode_generic_template0_into(mq, cx_state, (int)hgw, (int)hgh,
                                                      planes[i].data(), atx1, aty1,
                                                      -3, -1, 2, -2, -2, -2, false, skip);
                else
                    mq_encode_generic_template123_into(mq, cx_state, (int)hgw, (int)hgh,
                                                        planes[i].data(), htemplate,
                                                        atx1, aty1, false, skip);
            }
            mq_flush(mq);
            append(payload, mq.out.data(), mq.out.size());
        }
        append(d, payload.data(), payload.size());

        // 6.6.5.2 step 4: composite each cell's pattern onto the region
        // bitmap the same way DecodeImage does, so r.bitmap ends up
        // holding exactly what a decoder reconstructs -- DecodeImage
        // composites *every* cell unconditionally (it doesn't consult
        // HSKIP at all, only the bitplane decode does), so a skipped
        // cell's gval is still composited, just always as pattern 0 (see
        // the plane-building loop above).
        std::vector<uint8_t> canvas((size_t)ri.width * ri.height, hdefpixel ? 1 : 0);
        std::vector<uint8_t> patbuf((size_t)hpw * hph);
        for (uint32_t mg = 0; mg < hgh; mg++) {
            for (uint32_t ng = 0; ng < hgw; ng++) {
                bool skipped = henableskip && skipmask[(size_t)mg * hgw + ng];
                uint32_t pat_index = skipped ? 0 : (ng + mg) % hnumpats;
                for (uint32_t y = 0; y < hph; y++)
                    for (uint32_t x = 0; x < hpw; x++)
                        patbuf[(size_t)y * hpw + x] =
                            patseg->bitmap[(size_t)y * patseg->bw + pat_index * hpw + x];
                compose_bitmap(canvas, ri.width, ri.height, patbuf.data(), hpw, hph,
                               (int32_t)ng, (int32_t)mg, hcombop);
            }
        }
        r.bw = ri.width;
        r.bh = ri.height;
        r.bitmap = std::move(canvas);
    } else {
        // Coded gray-scale bitplanes, not modeled: no real pattern
        // dictionary available to place real patterns from.
        append_random_payload(d, 256);
    }

    printf("halftone-region handler (%zu bytes, %zu refs%s%s)\n", d.size(), refs.size(),
           real ? (hmmr ? ", real MMR content" : ", real arithmetic content") : "",
           henableskip ? ", HENABLESKIP" : "");
    r.data = d;
    r.refs = refs;
    r.colored = ri.colored;
    r.region_x = ri.x;
    r.region_y = ri.y;
    r.combop = ri.combop;
    return r;
}

// 7.4.6.1: generic region segment data header (Figure 47). Starts with
// the common region segment information field (7.4.1).
SegResult gen_segment_generic_region(const std::vector<GeneratedSegment> &)
{
    // 7.4.6.2: generic region segment flags (1 byte). Decide every flag
    // up front: real content (MMR, or any arithmetic GBTEMPLATE/EXTTEMPLATE
    // combination other than EXTTEMPLATE+TPGDON together -- see
    // mq_encode_generic_template0_ext()'s comment for why that pair has no
    // encoder) needs the declared region size to agree with what's
    // actually encoded, so a small cap; that one unmodeled combination
    // keeps the page-plausible 0x10000 cap with random payload, the same
    // as every other unmodeled shape elsewhere in this generator.
    bool mmr = (urand() & 1) != 0;
    uint8_t gbtemplate = mmr ? 0 : (uint8_t)(urand() % 4);
    // EXTTEMPLATE only has a defined AT-field layout when GBTEMPLATE is 0.
    bool exttemplate = (!mmr && gbtemplate == 0) && (urand() & 1);
    bool tpgdon = !mmr && (urand() & 1);
    bool arith_real = !mmr && !(exttemplate && tpgdon);

    // 128 (not some smaller value): needs to comfortably clear 32 pixels
    // even after gen_segment_region_info()'s own negative-X left clip
    // (up to 32 pixels) for a region composed onto the page to still span
    // multiple 32-bit words afterwards -- gen_text_region_real() already
    // proves 128x128 real content encodes/decodes fine at this size.
    RegionInfo ri = gen_segment_region_info(false, (mmr || arith_real) ? 128 : 0x10000);
    std::vector<uint8_t> d = ri.bytes;

    uint8_t flags = mmr ? 0x01 : 0x00;
    flags |= (uint8_t)(gbtemplate << 1);
    if (tpgdon)
        flags |= 0x08;
    if (exttemplate)
        flags |= 0x10;
    d.push_back(flags);

    // 7.4.6.3: generic region segment AT flags, present only if MMR == 0.
    // Half the time (when not EXTTEMPLATE, which has no defined nominal
    // layout), emit the spec's nominal AT positions instead of drawing
    // random ones -- see write_nominal_at_pixel()'s comment: that's the only
    // way a real decoder's optimized fixed-AT decode routine ever runs.
    AtPixel at1 = {}, at2 = {}, at3 = {}, at4 = {};
    AtPixel at_ext[12] = {};
    bool use_nominal_at = !exttemplate && (urand() & 1);
    if (!mmr) {
        int npairs = (gbtemplate == 0) ? (exttemplate ? 12 : 4) : 1;
        for (int i = 0; i < npairs; i++) {
            AtPixel at = use_nominal_at
                ? write_nominal_at_pixel(d, GB_NOMINAL_AT[gbtemplate][i].x,
                                          GB_NOMINAL_AT[gbtemplate][i].y)
                : write_primary_at_pixel(d);               // GBATXn/GBATYn
            if (exttemplate)
                at_ext[i] = at;
            else if (i == 0) at1 = at;
            else if (i == 1) at2 = at;
            else if (i == 2) at3 = at;
            else if (i == 3) at4 = at;
        }
    }

    std::vector<uint8_t> px;
    if (mmr || arith_real) {
        // Real coded content: a small synthetic bitmap, genuinely decodable
        // -- via T.6 (6.2.6) for MMR, or the MQ coder (6.2.5.7) for
        // arithmetic. GBTEMPLATE 0 has its own encoder (four AT pixels, a
        // 16-bit context) plus a separate one for EXTTEMPLATE=1 (twelve AT
        // pixels, a differently-arranged 16-bit context); 1-3 share one (a
        // single AT pixel, narrower contexts). Every non-ext encoder
        // mutates `px` in place when tpgdon is set, since a row it codes
        // as "typical" must actually equal the row above (6.2.5.7 step 3c)
        // -- so `px` (and r.bitmap below) end up holding what a decoder
        // reconstructs, not the checkerboard this seeds every row with.
        // The unknown-length terminator for immediate generic regions is
        // appended separately by gensegment().
        px.resize((size_t)ri.width * ri.height);
        for (uint32_t y = 0; y < ri.height; y++)
            for (uint32_t x = 0; x < ri.width; x++)
                px[(size_t)y * ri.width + x] = (uint8_t)((x ^ y) & 1);
        std::vector<uint8_t> coded =
            mmr ? mmr_encode((int)ri.width, (int)ri.height, px.data())
            : exttemplate
                ? mq_encode_generic_template0_ext(
                      (int)ri.width, (int)ri.height, px.data(), at_ext)
            : gbtemplate == 0
                ? mq_encode_generic_template0(
                      (int)ri.width, (int)ri.height, px.data(),
                      at1.x, at1.y, at2.x, at2.y, at3.x, at3.y, at4.x, at4.y,
                      tpgdon)
                : mq_encode_generic_template123(
                      (int)ri.width, (int)ri.height, px.data(), gbtemplate,
                      at1.x, at1.y, tpgdon);
        // For the unknown-length case (7.2.7/7.4.6.4), gensegment() places
        // its own terminator + row-count bytes immediately after this
        // segment's data ends, on the assumption that a real decode of
        // exactly GBW x GBH pixels stops right there. It doesn't: mq_flush()
        // emits its own ordinary MQ-coder flush overhead beyond the byte a
        // real decode actually needs (verified empirically -- a real decode
        // stopped 1-8+ bytes short of mq_encode_generic_template0()'s raw
        // output across sampled cases here), so gensegment()'s terminator
        // would land mid-flush-overhead, not at the real end, corrupting
        // everything after (including outright hangs -- this was the ~83%
        // failure rate a stress run found for unknown-length generic
        // regions). Trim to the exact byte count mq_finalize_generic()
        // measures via a faithful self-decode -- same fix as
        // mq_finalize_refinement() already applies for refinement regions.
        // Harmless for the known-length case too (still decodes correctly,
        // just slightly smaller), and EXTTEMPLATE isn't covered (a
        // different context formula mq_measure_generic_bytes() doesn't
        // model) -- unknown-length EXTTEMPLATE generic regions keep the
        // pre-existing bug for now.
        if (!mmr && !exttemplate) {
            coded = mq_finalize_generic((int)ri.width, (int)ri.height, gbtemplate,
                                         at1.x, at1.y, at2.x, at2.y, at3.x, at3.y, at4.x, at4.y,
                                         tpgdon, coded);
        }
        append(d, coded.data(), coded.size());
    } else {
        // Arithmetic-coded bitmap data, not modeled.
        append_random_payload(d, 256);
    }

    printf("generic-region handler (%zu bytes%s%s)\n", d.size(),
           mmr ? ", real MMR content" : arith_real ? ", real arithmetic content" : "",
           exttemplate ? ", EXTTEMPLATE" : "");
    SegResult r;
    r.data = d;
    // Known exactly only when this segment coded real content; the
    // random-payload branch leaves it empty so nothing refers to it.
    if (!px.empty()) {
        r.bw = ri.width;
        r.bh = ri.height;
        r.bitmap = std::move(px);
    }
    r.colored = ri.colored;
    r.region_x = ri.x;
    r.region_y = ri.y;
    r.combop = ri.combop;
    r.ext_template = exttemplate;
    r.mmr = mmr;
    // Every branch above codes the full declared height: the two real ones
    // encode exactly ri.height rows, and the random-payload branch stands
    // in for data that would. 7.4.6.4 permits a segment to carry fewer
    // rows than its region info field declares, but only if the row count
    // reports the number genuinely present.
    r.region_rows = ri.height;
    return r;
}

// 7.4.7.1: generic refinement region segment data header (Figure 52).
// Starts with the common region segment information field (7.4.1).
SegResult gen_segment_refinement_region(const std::vector<GeneratedSegment> &prior)
{
    // 7.4.7.4: may optionally refer to another region segment to select
    // GRREFERENCE; without one, the page buffer is used instead. Decided
    // before the region info field because 7.4.7.5 step 1 ties the choice
    // to this segment's own external combination operator, and because a
    // referred-to region also fixes this region's declared size (below).
    std::vector<uint32_t> refs;
    std::vector<uint32_t> region_pool;
    // Referred-to regions whose exact bitmap this generator knows, so it can
    // encode against the same GRREFERENCE the decoder will reconstruct.
    // Restricted to *intermediate* regions because only those survive as a
    // referable segment result -- an immediate region is composed straight
    // onto the page buffer and its own buffer discarded (7.4.7.5 step 5,
    // 8's page-decode procedure: "The other region segment must be a
    // previously occurring INTERMEDIATE region segment"; pdfium's own
    // ParseGenericRefinementRegion() enforces exactly this, rejecting any
    // referred-to segment whose type isn't 4/20/36/40 with a hard parse
    // failure before it even looks at content) -- and, among those, to the
    // types this generator codes real content for.
    std::vector<uint32_t> refbitmap_pool;
    for (const auto &seg : prior) {
        switch (seg.type) {
        case SEG_INTERMEDIATE_GENERIC: case SEG_INTERMEDIATE_GENERIC_REFINEMENT:
        case SEG_INTERMEDIATE_TEXT:
            if (!seg.bitmap.empty())
                refbitmap_pool.push_back(seg.number);
            [[fallthrough]];
        case SEG_INTERMEDIATE_HALFTONE:
            region_pool.push_back(seg.number);
            break;
        default:
            break;
        }
    }

    // Two GRREFERENCE sources are genuinely different decode paths, so split
    // runs between them rather than letting one crowd the other out: a
    // referred-to region's bitmap, and (7.4.7.4, no reference at all) the
    // page buffer. The third shape -- a reference this generator cannot
    // reproduce -- is what the random-payload fallback below stands in for.
    const GeneratedSegment *refseg = nullptr;
    if (!refbitmap_pool.empty() && (urand() & 1)) {
        // 7.4.7.4 permits exactly one referred-to intermediate region here,
        // and a decoder takes the reference from the first entry of the
        // referred-to list, so this is a single-element list by construction.
        refs = pick_refs(refbitmap_pool, 1, 1);
        for (const auto &seg : prior) {
            if (seg.number == refs[0]) {
                refseg = &seg;
                break;
            }
        }
    } else {
        refs = pick_refs(region_pool, 0, 1);
    }

    // 7.4.7.2: generic refinement region segment flags. Decide up front:
    // real content needs the declared region size to agree with what is
    // actually encoded, so a small cap; the fallback keeps the
    // page-plausible 0x10000 cap with random payload, mirroring
    // gen_segment_generic_region(). Both GRTEMPLATE values and both TPGRON
    // settings are genuinely encodable, and a referred-to region now
    // supplies a known GRREFERENCE, so the only shape left to the fallback
    // is a reference whose bitmap this generator never coded.
    uint8_t grtemplate = (uint8_t)(urand() % 2);
    bool tpgron = (urand() & 1) != 0;
    bool arith_real = refs.empty() || refseg;

    // With no referred-to region, the decoder builds GRREFERENCE out of the
    // page buffer itself -- page_->SubImage(x, y, w, h) in
    // ParseGenericRefinementRegion -- so this region can only code real
    // content if it knows both where it sits on the page and what is
    // already there. Exactly two placements make that knowable:
    //
    //  - Fully inside the page: the reference is the tracked g_page_bitmap
    //    rectangle. Width is held to a multiple of 32 and X to a byte
    //    boundary so SubImage's row copy moves whole bytes of real page
    //    content and its result's stride comes out exactly the region
    //    width -- no padding bits, which the per-pixel encoder here (and
    //    every other refinement caller) treats as 0 and would otherwise
    //    disagree about.
    //  - Entirely off the page: SubImage's own bounds check (x >= width_)
    //    returns early with an all-zero bitmap of the requested size. That
    //    needs no page tracking at all, so it stands in whenever the page
    //    state is unknown or the page is too small to place inside.
    //
    // Left to a free placement the region would usually land *partly* on
    // the page, where the reference is a mix of real content and
    // out-of-bounds zero padding that no invented pattern reproduces --
    // which is what made this path fail (and, at large declared sizes,
    // grind) before.
    uint32_t ref_force_w = 0, ref_force_h = 0;
    int64_t ref_force_x = INT64_MIN, ref_force_y = INT64_MIN;
    bool page_ref_inside = false;
    if (!refseg && refs.empty()) {
        uint32_t maxw = g_page_width < 128 ? g_page_width : 128;
        uint32_t maxh = g_page_height < 128 ? g_page_height : 128;
        if (g_page_state_known && maxw >= 32) {
            ref_force_w = 32 * (1 + urand() % (maxw / 32));
            ref_force_h = 1 + urand() % maxh;
            uint32_t xslots = (g_page_width - ref_force_w) / 8;
            ref_force_x = (int64_t)(8 * (urand() % (xslots + 1)));
            ref_force_y = (int64_t)(urand() % (g_page_height - ref_force_h + 1));
            page_ref_inside = true;
        } else {
            ref_force_w = 1 + urand() % 128;
            ref_force_h = 1 + urand() % 128;
            ref_force_x = (int64_t)g_page_width + (int64_t)(urand() % 1024);
            ref_force_y = (int64_t)(urand() % 0x10000);
        }
    }

    // 7.4.7.5 step 1: if this segment refers to no other region segment,
    // its external combination operator must be REPLACE. With a reference
    // the region must declare that reference bitmap's exact size, since the
    // decoder samples GRREFERENCE over this region's own extent (6.3.5.3).
    // 128, matching gen_segment_generic_region()'s own real-content cap and
    // for the same reason (headroom past a negative-X left clip) --
    // moot whenever a forced size below overrides it entirely.
    RegionInfo ri = gen_segment_region_info(/*force_replace=*/refs.empty(),
                                              arith_real ? 128 : 0x10000,
                                              refseg ? refseg->bw : ref_force_w,
                                              refseg ? refseg->bh : ref_force_h,
                                              ref_force_x, ref_force_y);
    std::vector<uint8_t> d = ri.bytes;

    uint8_t flags = grtemplate;
    if (tpgron)
        flags |= 0x02;
    d.push_back(flags);

    // 7.4.7.3: generic refinement region AT flags, present only if
    // GRTEMPLATE == 0. Half the time, use the nominal (-1,-1)/(-1,-1) pair
    // instead of drawing randomly -- pdfium's GRTEMPLATE0 decode only takes
    // its optimized fixed-AT path (DecodeTemplate0Opt) when both pixels
    // equal that nominal value; see write_nominal_at_pixel()'s comment.
    AtPixel at1 = {}, at2 = {};
    if (grtemplate == 0) {
        bool use_nominal_at = (urand() & 1) != 0;
        at1 = use_nominal_at ? write_nominal_at_pixel(d, -1, -1)
                              : write_primary_at_pixel(d);   // GRATX1/GRATY1
        at2 = use_nominal_at ? write_nominal_at_pixel(d, -1, -1)
                              : write_reference_at_pixel(d); // GRATX2/GRATY2
    }

    SegResult r;
    if (arith_real) {
        // Real arithmetic-coded content, either GRTEMPLATE, TPGRON on or
        // off, coded via the MQ coder (6.3.5.6).
        std::vector<uint8_t> cur((size_t)ri.width * ri.height);
        std::vector<uint8_t> ref((size_t)ri.width * ri.height);
        for (uint32_t y = 0; y < ri.height; y++)
            for (uint32_t x = 0; x < ri.width; x++)
                cur[(size_t)y * ri.width + x] = (uint8_t)((x ^ y) & 1);
        if (refseg) {
            // GRREFERENCE is the referred-to region's own bitmap, which the
            // decoder reconstructs from that segment -- so encoding against
            // it makes this region decode to exactly `cur`, rather than
            // merely decoding without error.
            ref = refseg->bitmap;
        } else if (page_ref_inside) {
            // GRREFERENCE is the page buffer's own contents under this
            // region, which the placement above put fully in bounds -- so
            // SubImage hands the decoder exactly this rectangle.
            for (uint32_t y = 0; y < ri.height; y++)
                for (uint32_t x = 0; x < ri.width; x++)
                    ref[(size_t)y * ri.width + x] =
                        g_page_bitmap[(size_t)((uint32_t)ri.y + y) * g_page_width
                                      + ((uint32_t)ri.x + x)];
        }
        // Otherwise the region sits off the page entirely and SubImage
        // returns an all-zero bitmap -- which `ref` already is.
        std::vector<uint8_t> coded = mq_encode_refinement(
            (int)ri.width, (int)ri.height, cur.data(), ref.data(),
            grtemplate, tpgron, at1.x, at1.y, at2.x, at2.y);
        // ParseGenericRefinementRegion() does `stream_->alignByte();
        // stream_->addOffset(2);` unconditionally right after a successful
        // Decode() -- the same fixed 2-byte skip the text-region-embedded
        // refinement case needs (see mq_finalize_refinement()'s comment).
        // Every prior test of this path stayed at or under 48x48 (the real
        // -content cap for both "no reference" and a generic-region
        // reference, which is itself capped at 48), where mq_flush()'s own
        // trailing bytes happened to be enough lookahead margin; a
        // text-region reference can now reach 128x128, and at that size
        // TPGRON's extra per-row Decode() call exhausts that margin before
        // every row is coded, making CJBig2_ArithDecoder::IsComplete()
        // trip mid-loop and Decode() return null. Measuring the real
        // consumption the same way closes that regardless of size.
        uint32_t rsize_unused;
        std::vector<uint8_t> final_bytes = mq_finalize_refinement(
            (int)ri.width, (int)ri.height, ref.data(), grtemplate,
            at1.x, at1.y, at2.x, at2.y, coded, &rsize_unused);
        append(d, final_bytes.data(), final_bytes.size());
        d.push_back(0xFF);
        d.push_back(0xFF);
        // mq_encode_refinement() rewrites `cur` where TPGRON skipped a
        // pixel, so it now holds exactly what a decoder reconstructs --
        // true for every real-content case here, since each one encoded
        // against the GRREFERENCE the decoder will really have (a
        // referred-to region's bitmap, the page rectangle under this
        // region, or the all-zero bitmap SubImage returns off-page). Carry
        // it up for both uses: as a later segment's GRREFERENCE, and as
        // what gensegment() composes onto its model of the page.
        r.bw = ri.width;
        r.bh = ri.height;
        r.bitmap = std::move(cur);
    } else {
        // Coded refinement bitmap data, not modeled.
        append_random_payload(d, 256);
    }

    printf("generic-refinement-region handler (%zu bytes, %zu refs%s)\n", d.size(), refs.size(),
           refseg ? ", real arithmetic content, real reference"
                  : arith_real ? ", real arithmetic content" : "");
    r.data = d;
    r.refs = refs;
    r.colored = ri.colored;
    r.region_x = ri.x;
    r.region_y = ri.y;
    r.combop = ri.combop;
    return r;
}

// 7.4.12: profiles segment - a count followed by that many 4-byte profile
// identification numbers.
SegResult gen_segment_profiles(const std::vector<GeneratedSegment> &)
{
    std::vector<uint8_t> d;
    uint32_t nprofiles = urand() % 8;
    put_be32(d, nprofiles);
    for (uint32_t i = 0; i < nprofiles; i++)
        put_be32(d, urand());
    printf("profiles handler (%zu bytes)\n", d.size());
    return { d, {} };
}

// 7.4.10: end of stripe segment data - a single four-byte value giving the
// Y coordinate of the stripe's end row.
SegResult gen_segment_end_of_stripe(const std::vector<GeneratedSegment> &)
{
    std::vector<uint8_t> d;
    put_be32(d, urand() % 0x10000);
    printf("end-of-stripe handler (%zu bytes)\n", d.size());
    return { d, {} };
}

// Annex B.2: code table (tables segment) structure. Flags select HTOOB and
// the bit widths HTPS/HTRS; HTLOW/HTHIGH bound the coded range; the table
// lines that follow are bit-packed (not byte-aligned), MSB-first per 5.4.1.
SegResult gen_segment_tables(const std::vector<GeneratedSegment> &)
{
    std::vector<uint8_t> d;

    bool htoob = (urand() & 1) != 0;
    uint8_t htrs = 1 + (uint8_t)(urand() % 8);   // B.2.1 field + 1

    // B.2 step 5: a decoder reads ordinary table lines in a loop driven by
    // CURRANGELOW, which starts at HTLOW and advances by 2^RANGELEN per
    // line, continuing only while CURRANGELOW < HTHIGH -- there is no
    // explicit line-count field. To make that loop stop exactly after the
    // `nlines` lines below (instead of reading past our data, or looping
    // far beyond it), every line but the last uses RANGELEN 0
    // (CURRANGELOW += 1); HTHIGH is then derived from where the last
    // line's RANGELEN leaves CURRANGELOW, rather than picked independently.
    // At least one line, so HTHIGH ends up strictly greater than HTLOW --
    // real decoders reject HTLOW >= HTHIGH (which nlines == 0 would give,
    // since CURRANGELOW would never advance).
    uint32_t nlines = 1 + urand() % 16;

    // B.3 assigns prefix codes canonically from the PREFLEN values, so the
    // set of lengths has to *be* a usable prefix code. An independent
    // random PREFLEN per line almost never is one, and with HTPS up to 8 a
    // field can name a length of 255 -- far past the 32 B.3's accumulator
    // can represent, which real decoders reject outright (pdfium's
    // HuffmanAssignCode bails on the int32 overflow), taking with it every
    // symbol-dictionary and text-region selector that referred to this
    // table. Give every line the same length L with 2^L >= line count
    // instead: Kraft is satisfied with room to spare, and B.3 then hands
    // line i the plain L-bit code i -- the same device write_symbol_id_table()
    // uses for symbol IDs. The lines are the `nlines` ordinary ones plus
    // the lower and upper range lines, plus the OOB line when HTOOB is set.
    uint32_t total_lines = nlines + 2 + (htoob ? 1 : 0);
    uint8_t preflen = 1;
    while ((1u << preflen) < total_lines)
        preflen++;

    // HTPS is the width of every PREFLEN field, so it has to be wide enough
    // to carry `preflen` itself; draw only from the widths that are.
    uint8_t htps_min = 1;
    while ((1u << htps_min) - 1 < preflen)
        htps_min++;
    uint8_t htps = htps_min + (uint8_t)(urand() % (8 - htps_min + 1));

    // B.2.1: code table flags (1 byte).
    uint8_t flags = htoob ? 0x01 : 0x00;
    flags |= (uint8_t)((htps - 1) << 1);
    flags |= (uint8_t)((htrs - 1) << 4);
    d.push_back(flags);

    int32_t htlow = (int32_t)(urand() % 0x10000);   // B.2.2: HTLOW

    // The last line's RANGELEN must still fit in HTRS bits once written
    // (otherwise bw_put_bits() truncates it and the decoder would read
    // back a different value than what HTHIGH was computed from), and
    // stay small enough that 1 << RANGELEN can't approach int32 overflow.
    uint32_t rangelen_cap = (1u << htrs) - 1;
    if (rangelen_cap > 20)
        rangelen_cap = 20;
    BitWriter bw;
    int32_t currangelow = htlow;
    uint8_t last_rangelen = 0;
    for (uint32_t i = 0; i < nlines; i++) {
        uint8_t rangelen = (i + 1 == nlines) ? (uint8_t)(1 + urand() % rangelen_cap) : 0;
        if (i + 1 == nlines)
            last_rangelen = rangelen;
        bw_put_bits(bw, preflen, htps);      // PREFLEN
        bw_put_bits(bw, rangelen, htrs);     // RANGELEN
        currangelow += (1 << rangelen);
    }
    int32_t hthigh = currangelow;            // B.2.3: HTHIGH

    put_be32(d, (uint32_t)htlow);
    put_be32(d, (uint32_t)hthigh);

    // B.2 steps 6-9: lower and upper range table lines (HTPS bits each).
    bw_put_bits(bw, preflen, htps);       // LOWPREFLEN
    bw_put_bits(bw, preflen, htps);       // HIGHPREFLEN
    // B.2 step 10: out-of-band table line, only if HTOOB.
    if (htoob)
        bw_put_bits(bw, preflen, htps);   // OOBPREFLEN
    bw_finish(bw);
    // Real decoders check each read as "boffset + width < available bits"
    // rather than "<=", so a bit-packed buffer with zero slack after
    // byte-padding (i.e. the exact bit count already lands on a byte
    // boundary) can be one bit short for the very last read even though
    // every bit written was consumed exactly as intended. Pad with a spare
    // byte so there's always slack left over.
    bw.bytes.push_back(0);
    append(d, bw.bytes.data(), bw.bytes.size());

    printf("tables handler (%zu bytes, nlines=%u preflen=%u htps=%u htlow=%d hthigh=%d)\n",
           d.size(), nlines, preflen, htps, htlow, hthigh);
    SegResult r;
    r.data = d;
    // The last ordinary line (RANGELOW = HTLOW + nlines - 1) is the one
    // line a later real-content generator can encode against with zero
    // offset bits, no matter what shape the rest of this table takes --
    // see GeneratedSegment::table_rows's comment. Its code is nlines - 1:
    // canonical codes are assigned in table order (ordinary lines first),
    // and every line here shares one PREFLEN, so B.3 hands out plain
    // sequential codes the same way write_symbol_id_table() does.
    r.table_rows.push_back({ (int32_t)(nlines - 1), (int32_t)preflen,
                              (int32_t)last_rangelen, htlow + (int32_t)(nlines - 1), 0 });
    return r;
}

// 7.4.16.1: colour palette segment data header (Figure 58) plus the coded
// palette values described in 7.4.16.2/6.8.4.
SegResult gen_segment_colour_palette(const std::vector<GeneratedSegment> &)
{
    std::vector<uint8_t> d;

    // 7.4.16.1.1: colour palette flags (1 byte). Bit 0 (continuation) is
    // left 0: this generator always emits a single flags byte. Bits 1-4
    // colour space (0-2 defined; 3-15 reserved). Bits 5-7 reserved.
    uint8_t colour_space = (uint8_t)(urand() % 3);
    d.push_back((uint8_t)(colour_space << 1));

    uint8_t cpncomp = (uint8_t)(1 + (urand() % 4));       // CPNCOMP: 1..255
    d.push_back(cpncomp);

    static const uint8_t complens[3] = { 1, 2, 4 };
    uint8_t cpcomplen = complens[urand() % 3];            // CPCOMPLEN: 1, 2 or 4
    d.push_back(cpcomplen);

    uint32_t cpnvals = urand() % 64;                      // CPNVALS: kept small
    put_be32(d, cpnvals);

    // Remainder: CPNVALS * CPNCOMP colour component values, CPCOMPLEN bytes each.
    uint32_t payload_len = cpnvals * cpncomp * cpcomplen;
    size_t off = d.size();
    d.resize(off + payload_len);
    fill_random_pattern(d.data() + off, payload_len);

    printf("colour-palette handler (%zu bytes)\n", d.size());
    return { d, {} };
}

SegResult gensegmentdata(uint8_t segment_type, uint32_t max_len, const std::vector<GeneratedSegment> &prior)
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

    // Unimplemented (null) handlers are skipped: fall back to random data
    // with no refs.
    if (handler != nullptr)
        return handler(prior);

    uint32_t len = urand() % (max_len + 1);
    std::vector<uint8_t> data(len);
    fill_random_pattern(data.data(), len);
    printf("segment data generated (%u bytes)\n", len);
    return { data, {} };
}

// forced_type: SegmentType to use instead of picking one at random, or -1.
// Page information, end of page, end of file, and profiles are deliberately
// excluded from the random pool below and only ever produced via
// forced_type, so that callers can guarantee exactly one page information
// and end of page segment (7.4.8, 7.4.9), and that any profiles segment
// is placed and associated correctly (7.4.12).
// forced_page: page association to write instead of picking one at random,
// or -1. See gensegmentheader().
// forced_number: segment number to use instead of drawing the next one
// from g_next_segment_number, or -1. Lets a caller reserve a low number
// (bumping the counter) before generating content, then build that
// segment's bytes afterward once it can see that content in `prior` — see
// how main() builds the page information segment.
std::vector<std::vector<uint8_t> *> gensegment(int forced_type = -1, int32_t forced_page = -1,
                                                int64_t forced_number = -1)
{
    static const uint8_t types[] = {
        SEG_SYMBOL_DICTIONARY, SEG_INTERMEDIATE_TEXT, SEG_IMMEDIATE_TEXT,
        SEG_IMMEDIATE_LOSSLESS_TEXT, SEG_PATTERN_DICTIONARY,
        SEG_INTERMEDIATE_HALFTONE, SEG_IMMEDIATE_HALFTONE,
        SEG_IMMEDIATE_LOSSLESS_HALFTONE, SEG_INTERMEDIATE_GENERIC,
        SEG_IMMEDIATE_GENERIC, SEG_IMMEDIATE_LOSSLESS_GENERIC,
        SEG_INTERMEDIATE_GENERIC_REFINEMENT, SEG_IMMEDIATE_GENERIC_REFINEMENT,
        SEG_IMMEDIATE_LOSSLESS_GENERIC_REFINEMENT,
        SEG_END_OF_STRIPE,
        SEG_TABLES, SEG_COLOUR_PALETTE, SEG_EXTENSION
    };
    uint8_t type = (forced_type >= 0)
                       ? (uint8_t)forced_type
                       : types[urand() % (sizeof(types) / sizeof(types[0]))];

    uint32_t segment_number = (forced_number >= 0) ? (uint32_t)forced_number : g_next_segment_number++;

    // 7.4.9/7.4.11: end-of-page and end-of-file segments carry no data.
    // 7.4.10: end-of-stripe does carry a 4-byte end-row value (see
    // gen_segment_end_of_stripe()).
    bool has_data = type != SEG_END_OF_PAGE && type != SEG_END_OF_FILE;

    std::vector<uint8_t> data;
    std::vector<uint32_t> refs;
    bool colored = false;
    int combop = -1;
    uint32_t num_symbols = 0;
    bool ext_template = false;
    bool mmr = false;
    uint32_t region_rows = 0;
    uint32_t bw = 0, bh = 0;
    std::vector<uint8_t> bitmap;
    std::vector<ExportedSymbol> symbols;
    std::vector<StdHuffLine> table_rows;
    uint32_t hdpw = 0;
    int32_t region_x = 0, region_y = 0;
    if (has_data) {
        SegResult r = gensegmentdata(type, 256, g_prior_segments);
        data = std::move(r.data);
        refs = std::move(r.refs);
        colored = r.colored;
        combop = r.combop;
        num_symbols = r.num_symbols;
        ext_template = r.ext_template;
        mmr = r.mmr;
        region_rows = r.region_rows;
        bw = r.bw;
        bh = r.bh;
        bitmap = std::move(r.bitmap);
        symbols = std::move(r.symbols);
        table_rows = std::move(r.table_rows);
        hdpw = r.hdpw;
        region_x = r.region_x;
        region_y = r.region_y;
    }
    uint32_t data_len = (uint32_t)data.size();

    // 8.2 step 5a: an immediate *direct* region segment's decoded bitmap is
    // combined straight into the page buffer using that region's own
    // external combination operator (an intermediate one goes to an
    // auxiliary buffer instead, and never reaches the page on its own).
    // Mirroring that here keeps g_page_bitmap in step with what a decoder
    // holds, which is what lets a later reference-less refinement region
    // encode against the page it will really sample (7.4.7.4). A region
    // whose content this generator did not code exactly cannot be
    // modelled -- and, carrying random payload, fails to decode and takes
    // the rest of the file with it anyway -- so it just retires the model.
    if (is_immediate_direct_region(type)) {
        if (g_page_state_known && !bitmap.empty() && combop >= 0) {
            compose_bitmap(g_page_bitmap, g_page_width, g_page_height,
                            bitmap.data(), bw, bh, region_x, region_y, (uint8_t)combop);
        } else {
            g_page_state_known = false;
        }
    }

    // 7.2.7: an immediate generic region *may* declare an unknown data
    // length, recoverable only by scanning for the trailer appended below.
    // Under random-access organization that is unusable: D.2 puts every
    // segment header before any segment data, so a reader walking the
    // header block has no length for this segment and cannot locate the
    // start of any later segment's data -- the file stops being walkable
    // at all. Keep lengths known there, and make it a coin flip elsewhere
    // so the (far more common) known-length shape is exercised too, which
    // an unconditional 0xFFFFFFFF never allowed.
    bool unknown_len = type == SEG_IMMEDIATE_GENERIC &&
                       g_organisation != ORG_RANDOM_ACCESS && (urand() & 1);

    if (unknown_len) {
        // With an unknown length, 7.4.6.4 says the data part ends with a
        // terminator plus a 4-byte row count (the number of rows *actually*
        // encoded, no greater than the declared bitmap height -- comes from
        // the handler that made that choice; re-deriving it by re-reading
        // the region info field out of `data` would silently decay into
        // nonsense the moment that layout changed). But *where* that
        // terminator has to sit is dictated by what a real decoder skips
        // after decoding, and the two coding methods disagree:
        //  - Arithmetic: ParseGenericRegion() does stream_->addOffset(2)
        //    after decode, unconditionally skipping 2 bytes before this
        //    field's own +4 (below). gen_segment_generic_region() already
        //    places its arithmetic-coded bytes' *end* at the exact byte a
        //    real decode stops at (mq_finalize_generic(), which measures
        //    that via a faithful self-decode -- mq_flush()'s own raw output
        //    is NOT that position, see its call site's comment), so an
        //    explicit 0xFF 0xAC there lines up with those 2 skipped bytes.
        //  - MMR: ParseGenericRegion() does only stream_->alignByte() after
        //    StartDecodeMMR() -- no 2-byte skip at all. FaxG4Decode's own
        //    bit-position tracking already lands exactly at the true
        //    byte-aligned end of the real T.6 data, so writing an extra
        //    "0x00 0x00" marker there (as 7.4.6.4's terminator convention
        //    literally describes) inserts 2 bytes nothing ever skips,
        //    misaligning the row count and everything after it.
        if (!mmr) {
            static const uint8_t term_arith[2] = { 0xFF, 0xAC };
            append(data, term_arith, 2);
        }
        put_be32(data, region_rows);
        data_len = (uint32_t)data.size();
    }

    std::vector<uint8_t> *hdr = new std::vector<uint8_t>(
        gensegmentheader(type, segment_number, refs, data_len, &g_segment_len, forced_page,
                         unknown_len));

    // Make this segment visible to later gensegment() calls as a possible
    // referent, now that its own refs (which could only point to earlier
    // segments) are settled.
    g_prior_segments.push_back({ segment_number, type,
                                  (uint32_t)(forced_page >= 0 ? forced_page : 0),
                                  colored, combop, (uint32_t)refs.size(), num_symbols, ext_template,
                                  bw, bh, std::move(bitmap), std::move(symbols), std::move(table_rows), hdpw });

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

static void push_segment(std::vector<std::vector<uint8_t> *> &seg)
{
    header_streams.push_back(seg[0]);
    data_streams.push_back(seg[1]);
}

// Master init: everything that must happen before generation starts.
void init_all(void)
{
    urand_init();
    init_seg_handlers();
    mmr_check_tables();
}

int main(int argc, char **argv)
{
    // D.4: the file header (which starts with the 8-byte ID magic) is
    // present for the sequential and random-access organizations, and
    // absent for embedded. --header/--no-header pin that choice instead of
    // leaving it to choose_organisation()'s random pick, so callers that
    // need a specific shape (e.g. a decoder harness's own with/without-magic
    // test cases) don't have to keep regenerating until they get lucky.
    enum { HEADER_RANDOM, HEADER_FORCE_ON, HEADER_FORCE_OFF } header_mode = HEADER_RANDOM;
    const char *out_path = "out.jb2";
    // --dump-page writes the page this run believes it built, in exactly
    // the P4 PBM layout jbig2dec emits, so the two can be compared
    // byte-for-byte: a decoder that reconstructs a different page than the
    // generator intended is a bug even when it reports success. Nothing is
    // written when the page state is unknown (see g_page_state_known).
    const char *dump_page_path = nullptr;
    bool out_path_set = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--header") == 0) {
            header_mode = HEADER_FORCE_ON;
        } else if (strcmp(argv[i], "--no-header") == 0) {
            header_mode = HEADER_FORCE_OFF;
        } else if (strcmp(argv[i], "--dump-page") == 0 && i + 1 < argc) {
            dump_page_path = argv[++i];
        } else if (argv[i][0] == '-' || out_path_set) {
            fprintf(stderr, "Usage: %s [out_path] [--header|--no-header] [--dump-page <path.pbm>]\n",
                    argv[0]);
            return 1;
        } else {
            out_path = argv[i];
            out_path_set = true;
        }
    }

    init_all();
    Organization org;
    bool write_header;
    switch (header_mode) {
    case HEADER_FORCE_ON:
        // Both header-bearing organizations stay in play, chosen the same
        // way choose_organisation() would between them.
        org = (urand() & 1) ? ORG_SEQUENTIAL : ORG_RANDOM_ACCESS;
        write_header = true;
        break;
    case HEADER_FORCE_OFF:
        // --no-header only omits the file header's magic/flags/page-count
        // bytes; it leaves the organization choice (and so the segment/data
        // layout assemble_org_order() produces) exactly as random as
        // HEADER_RANDOM's. Some setups want a headerless stream in an
        // otherwise-arbitrary organization -- e.g. random-access's
        // headers-then-data shape without the header that would normally
        // be the only way a decoder learns to expect that shape.
        org = choose_organisation();
        write_header = false;
        break;
    case HEADER_RANDOM:
    default:
        org = choose_organisation();
        write_header = (org != ORG_EMBEDDED);
        break;
    }
    g_organisation = org;
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

    // Settle the page geometry before any content is generated, so region
    // placement has a page to aim at and g_page_bitmap can track what a
    // decoder's page buffer holds -- gen_segment_page_info() below just
    // reports what this chose. (Its segment number is reserved further up
    // and its bytes are built last, so the page information segment still
    // lands first in the file, per 7.4.8.)
    choose_page_geometry();

    // 7.4.12: if a profiles segment is present, it must be the very first
    // segment of the data stream, and must not be associated with any page.
    if (urand() & 1) {
        std::vector<std::vector<uint8_t> *> profiles = gensegment(SEG_PROFILES, 0);
        push_segment(profiles);
    }

    // Build one coherent page (page number 1). 7.4.8 requires the page
    // information segment to be the first segment associated with the
    // page, so its segment number is reserved here, before any content —
    // but its actual bytes are built after the content loop below (once
    // gen_segment_page_info() can see that content in g_prior_segments and
    // derive flags like "might contain refinements" from it), then spliced
    // back to this position.
    size_t page_info_pos = header_streams.size();
    uint32_t page_info_number = g_next_segment_number++;

    // Content-segment count. Reference-shape coverage is bounded by how
    // many earlier segments of a given type exist to refer to, and with
    // 0-3 content segments spread across ~18 types the per-type candidate
    // pools were nearly always empty or singleton -- "oldest" and "newest"
    // then name the same segment and pick_refs()'s multi-reference
    // strategy is never even offered. Draw from two bands: small files
    // stay common (they exercise the empty-pool and no-reference paths)
    // while larger ones make the pools deep enough for those strategies
    // to describe genuinely different shapes.
    size_t ncontent = (urand() & 1) ? urand() % 4 : 4 + urand() % 29;   // 0..3 or 4..32
    for (size_t i = 0; i < ncontent; i++) {
        std::vector<std::vector<uint8_t> *> seg = gensegment(-1, 1);
        push_segment(seg);
    }

    std::vector<std::vector<uint8_t> *> page_info =
        gensegment(SEG_PAGE_INFORMATION, 1, page_info_number);
    header_streams.insert(header_streams.begin() + page_info_pos, page_info[0]);
    data_streams.insert(data_streams.begin() + page_info_pos, page_info[1]);

    // 7.4.9: each page must have exactly one end of page segment associated
    // with it, and it must be the last segment associated with that page.
    std::vector<std::vector<uint8_t> *> end_of_page = gensegment(SEG_END_OF_PAGE, 1);
    push_segment(end_of_page);

    // D.2: random-access organization requires the file's last segment to
    // be an end of file segment (7.4.11), unassociated with any page.
    if (org == ORG_RANDOM_ACCESS) {
        std::vector<std::vector<uint8_t> *> end_of_file = gensegment(SEG_END_OF_FILE, 0);
        push_segment(end_of_file);
    }

    // D.4.2 bits 2-3: "12 adaptive template pixels used" and "coloured
    // region present" must match whether any generic region actually set
    // EXTTEMPLATE, and whether any region segment actually set COLEXTFLAG
    // (the same colour signal gen_segment_page_info() used for the page
    // information segment's own "might contain coloured segment" bit).
    // Content is fully known now, so the file header (which must be the
    // file's first bytes) is built last and relies on
    // assemble_org_order()/serialize_out() running afterward to place it
    // correctly.
    k.use_12_AT = false;
    k.colored_region = false;
    for (const auto &seg : g_prior_segments) {
        if (seg.ext_template)
            k.use_12_AT = true;
        if (seg.colored)
            k.colored_region = true;
    }
    genheader(org, k, 1, write_header);   // main() builds exactly one page (page number 1)

    assemble_org_order(org);

    serialize_out(stream.data(), stream.size(), out_path);

    // --dump-page: the page this run believes it built, in jbig2dec's own
    // P4 PBM layout. Bits come out inverted because pdfium's decoder flips
    // its whole output buffer once decoding finishes (jbig2_decoder.cpp's
    // `pix = ~pix`, PDF's ImageMask convention), and the bits past the page
    // width carry the default pixel value Fill() left there, inverted the
    // same way -- both are part of matching the file jbig2dec writes.
    if (dump_page_path && g_page_state_known) {
        FILE *pf = fopen(dump_page_path, "wb");
        if (pf) {
            fprintf(pf, "P4\n%u %u\n", g_page_width, g_page_height);
            size_t row_bytes = (g_page_width + 7) / 8;
            std::vector<uint8_t> row(row_bytes);
            for (uint32_t y = 0; y < g_page_height; y++) {
                // Padding bits past the page width keep whatever Fill()
                // put there (the default pixel), and the decoder's final
                // whole-buffer inversion flips them too.
                row.assign(row_bytes, g_page_default_pixel ? 0x00 : 0xFF);
                for (uint32_t x = 0; x < g_page_width; x++) {
                    uint8_t m = (uint8_t)(0x80 >> (x % 8));
                    if (!g_page_bitmap[(size_t)y * g_page_width + x])
                        row[x / 8] |= m;
                    else
                        row[x / 8] &= (uint8_t)~m;
                }
                fwrite(row.data(), 1, row_bytes, pf);
            }
            fclose(pf);
        }
    }

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
