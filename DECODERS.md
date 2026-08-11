# Decoders, oracles, and known divergences

Register of places where conforming-ish JBIG2 implementations provably disagree.
These are recorded rather than "fixed": a spot where two independent decoders
produce different output from the same bytes is something to generate *for*
deliberately, not just work around.

Code sites carry a matching `AMBIGUITY(<name>)` comment.

## Decoders available

| Path | What | Role |
|---|---|---|
| `jbig2dec/` | PDFium's `CJBig2_Context` behind a small CLI harness | primary oracle |
| `alt/jbig2dec1/` | Ghostscript jbig2dec (newer, `./autogen.sh && ./configure && make`) | second oracle |
| `alt/jbig2dec/` | Ghostscript jbig2dec 0.11 | superseded by the above |
| `alt/jbig2enc/` | Google's encoder (leptonica) | not a decoder; `src/jbig2arith.cc` cross-checks MQ logic |
| `alt/jbig2enc-rs/` | Rust encoder | unexamined |

The generator's own `--dump-page` is a third, independent opinion: the page it
believes it built, in jbig2dec's P4 PBM layout.

### Polarity

pdfium's harness emits the page **inverted** (`Jbig2Decoder::StartDecode` does
`pix = ~pix` over the whole buffer). `--dump-page` is written in that same
polarity, so model-vs-pdfium is a plain `cmp`. Ghostscript jbig2dec emits true
polarity — invert one side, and mask each row's padding bits, before comparing.

### Ground truth

`alt/jbig2dec1/annex-h.jbig2` is the **T.88 Annex H** canonical test sequence.
pdfium and Ghostscript jbig2dec agree on its page 1 (64x56) **bit for bit**
after the inversion above, which validates both decoders and the comparison
methodology itself. jbig2dec emits all of its pages; the pdfium harness emits
only page 1.

## Divergences

### AMBIGUITY(mmr-unknown-length) — unknown-length MMR generic region

7.4.6.4 terminates an unknown-length (`0xFFFFFFFF`) generic region with a
2-byte marker then a 4-byte row count. Ghostscript jbig2dec implements this
literally (`jbig2.c:376`): it **scans forward** from byte 18 for the marker
(`00 00` MMR, `FF AC` arithmetic), reads the row count after it, and derives
the segment's `data_length` from where the scan stopped. pdfium's MMR path
does only `alignByte()` after `StartDecodeMMR()` — no 2-byte skip — so it
expects the row count immediately after the data.

Because the declared length is unknown, both decoders locate every *later*
segment relative to this point, so the 2-byte disagreement desyncs the whole
rest of the file. **There is no encoding that satisfies both.** Measured over
150 unknown-length-MMR files each way:

| | pdfium | jbig2dec | row-count errors |
|---|---|---|---|
| no marker (what we emit) | 150/150 | 74/150 | 33 |
| marker per 7.4.6.4 | 63/150 | 77/150 | 0 |

jbig2dec is the one reading the spec correctly; we emit the non-conformant
form to keep the primary oracle at full coverage on this path. The 33 errors
are jbig2dec's scan locking onto the `00 00` that begins our own big-endian
row count.

The arithmetic case has no such conflict — pdfium *does* skip 2 bytes there,
so we write `FF AC` and both decoders are happy.

### Refinement reference is not subsetted (jbig2dec defect)

`alt/jbig2dec1/jbig2_refinement.c:480` carries its own
`/* TODO: subset the image if appropriate */`. When a refinement region segment
refers to no intermediate region, 7.4.7.2 makes GRREFERENCE the page rectangle
*under the region*; jbig2dec instead passes the **whole page** with
`GRREFERENCEDX = GRREFERENCEDY = 0`. pdfium does `SubImage(x, y, w, h)`.

Measured on forced immediate-refinement files, comparing model / pdfium /
jbig2dec three ways:

- 416 comparable files: **~48% divergence, uniform** across all four
  (TPGRON, GRTEMPLATE) combinations — it is not template-specific.
- Trigger is the **page default pixel**: `defpix=0` → 109 agree, 0 diverge;
  `defpix=1` → 12 agree, 113 diverge.
- Differing rows are always the region's rows *minus the first*.

In every such case **our model and pdfium agree and jbig2dec differs**, so this
is jbig2dec's defect, not ours. Consequence: **jbig2dec is not a usable oracle
for refinement regions.** The exact internal path from "un-subsetted reference"
to "only diverges when defpix=1" was not fully isolated — the trigger is
confirmed empirically, the mechanism is inferred from their TODO.

## Normative tables, cross-checked

### Annex E arithmetic coder state table (Table E.1)

**Identical** across our generator, PDFium, Ghostscript jbig2dec (both
versions) — all 47 states, comparing (Qe, NMPS, NLPS, SWITCH) after
normalizing each project's packing. Ours is packed `nlps|switch : nmps : Qe`
in a `uint32_t` and was taken from the ITU sample software's `QeIndexTable`;
it matches the others exactly.

The two encoders (`jbig2enc`, `jbig2enc-rs`) carry **46** states, not 47, laid
out as 46x2 for the two MPS polarities. That is safe rather than wrong: state
46 is reachable only by starting there, and JBIG2 initializes every context to
state 0, so an encoder can never enter it. States 0..45 match exactly.

### Standard Huffman tables B.1-B.15

Compared as multisets of (PREFLEN, RANGELEN, RANGELOW) against PDFium's
`kTableLine*` and jbig2dec's `jbig2_huffman_lines_*`. All 15 now agree across
all three.

They did not before: **Table B.12 carried the wrong PREFLEN on two rows** --
`VAL 6...7` and `VAL 8...9` were recorded as 6 and 7 where the spec (and both
decoders) say 5 and 6. The prefix *codes* were right (`0x1d` = `11101`,
`0x3c` = `111100`), so the effect would have been emitting a 5-bit code padded
to 6 bits and desyncing the decoder mid-stream. Latent at the time it was
found: both DT call sites encode the constant 1, so those rows are
unreachable, and a same-seed A/B over 400 seeds with `dt_table_sel` forced to
B.12 produced byte-identical output. It would have bitten the moment a DT
value reached 6.

A canonical-code validator now covers this class of bug: for each table, sort
lines by (PREFLEN, code) and require the first code of each length to equal
`(prev_code + 1) << (len - prev_len)`. B.12 was the only violation.

### SLTP / TPGDON reused contexts — *not* a table mismatch

Previously recorded as a "spec vs PDFium" divergence. It is not one. T.88
6.2.5.7 says of the CONTEXT value:

> "The order of this gathering is not standardized, but shall be consistent
> and independent of the location of the AT pixels."

So the SLTP context number is **implementation-defined**, valid as long as the
implementation is self-consistent. The spec's own worked example gathers in
reading order and gives `GB0011100101` for GBTEMPLATE 2 — that is `0x00E5`,
PDFium's value.

| Implementation | GBTEMPLATE 0/1/2/3 |
|---|---|
| PDFium | `0x9B25 0x0795 0x00E5 0x0195` |
| jbig2dec (both) | `0x9B25 0x0795 0x00E5 0x0195` |
| jbig2enc, jbig2enc-rs | `0x9B25` (template 0) |
| ITU sample software | `0xC395 0x0795 0x0271 0x02C5` |

Five implementations use reading order; the ITU sample software uses its own
gathering order and is internally consistent. Since an encoder must match its
decoder's convention, ours must stay on the reading-order set. PDFium's
refinement TPGRON contexts are `0x0010` (GRTEMPLATE 0) and `0x0008`
(GRTEMPLATE 1), which is what we encode against.

Consequence for the record: EXTTEMPLATE+TPGDON is unverifiable because PDFium
has no EXTTEMPLATE support at all, **not** because of SLTP constants.

## What blocks jbig2dec as an oracle (not our bugs)

Roughly 53% of generated files never reach segment parsing under jbig2dec:

- **File header Amendment 2/3 flags.** `0x04` ("12 adaptive template pixels",
  i.e. EXTTEMPLATE) and `0x08` ("colored region segments") are both fatal-NYI
  in jbig2dec (`jbig2.c:303`, `:307`). We set them correctly — they are derived
  from actual content at `header.cpp:5578` — jbig2dec just doesn't implement
  those amendments.
- **`unsupported image coordinates`.** Our deliberate off-page and negative
  region placements. pdfium clips; jbig2dec refuses.
- **`unhandled segment type 'intermediate generic region' (NYI)`.**

Also: jbig2dec keeps writing a partially-composed page *after* a fatal
("treating as end of file"). Comparing that partial output against pdfium's
clipped-and-continued output measures error-recovery policy, not decode
correctness — **filter on a clean stderr** (no `FATAL`, no `WARNING`) before
comparing. Doing so took the observed disagreement rate from 12.7% to 3.8%,
and the 3.8% residual was entirely the refinement defect above.

## Harness

`tools/three_way_oracle.py <count> <workdir>` — generates with `--dump-page`,
decodes with both decoders, normalizes polarity and padding, and classifies
each file as all-agree / pdfium-wrong / jbig2dec-wrong / model-wrong /
all-differ. Only files both decoders accept cleanly are counted.
