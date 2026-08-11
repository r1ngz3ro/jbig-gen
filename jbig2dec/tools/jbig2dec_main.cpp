// Copyright 2024 The PDFium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Standalone JBIG2 decoder CLI built on PDFium's JBIG2 codec.
//
// Usage: jbig2dec <input.jb2> [output.pbm]
//
// Decodes a JBIG2 file (ISO/IEC 14492, sequential or random-access
// organization) to a PBM (P4) image. Width/height are read from the page-info
// segment, which is required for PDFium's Jbig2Decoder::StartDecode() to
// allocate the destination bitmap.
#include <stdint.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/fxcodec/jbig2/jbig2_decoder.h"
#include "core/fxcodec/jbig2/jbig2_document_context.h"
#include "core/fxcrt/span.h"

namespace {

// D.4.1: the 8-byte ID string is the same for every organization -- there
// is no alternate form. Organization is signalled by bit 0 of the 1-byte
// flags field that follows it (D.4.2), not by the ID string itself.
constexpr uint8_t kJBIG2FileHeaderId[] = {0x97, 0x4a, 0x42, 0x32,
                                          0x0d, 0x0a, 0x1a, 0x0a};

enum class Organization { kSequential, kRandomAccess, kEmbedded };

struct ByteReader {
  const uint8_t* data;
  size_t size;
  size_t pos = 0;

  bool ReadBytes(size_t n, const uint8_t** out) {
    if (pos + n > size) {
      return false;
    }
    *out = data + pos;
    pos += n;
    return true;
  }

  bool ReadU16(uint16_t* value) {
    const uint8_t* bytes = nullptr;
    if (!ReadBytes(2, &bytes)) {
      return false;
    }
    *value = static_cast<uint16_t>((bytes[0] << 8) | bytes[1]);
    return true;
  }

  bool ReadU32(uint32_t* value) {
    const uint8_t* bytes = nullptr;
    if (!ReadBytes(4, &bytes)) {
      return false;
    }
    *value = (static_cast<uint32_t>(bytes[0]) << 24) |
             (static_cast<uint32_t>(bytes[1]) << 16) |
             (static_cast<uint32_t>(bytes[2]) << 8) |
             static_cast<uint32_t>(bytes[3]);
    return true;
  }

  bool Skip(size_t n) {
    if (pos + n > size) {
      return false;
    }
    pos += n;
    return true;
  }
};

// Mirrors CJBig2_Context::ParseSegmentHeader() in jbig2_context.cpp.
bool ParseSegmentHeader(ByteReader* reader,
                        uint32_t* segment_type,
                        uint32_t* data_length) {
  uint32_t segment_number = 0;
  if (!reader->ReadU32(&segment_number)) {
    return false;
  }
  const uint8_t* flags_byte = nullptr;
  if (!reader->ReadBytes(1, &flags_byte)) {
    return false;
  }
  const uint8_t flags = flags_byte[0];

  uint32_t referred_count = 0;
  const uint8_t first_count_byte =
      reader->pos < reader->size ? reader->data[reader->pos] : 0;
  if ((first_count_byte >> 5) == 7) {
    uint32_t raw_count = 0;
    if (!reader->ReadU32(&raw_count)) {
      return false;
    }
    referred_count = raw_count & 0x1fffffff;
    if (referred_count > 64) {
      return false;
    }
    if (!reader->Skip((1 + referred_count + 7) / 8)) {
      return false;
    }
  } else {
    const uint8_t* count_byte = nullptr;
    if (!reader->ReadBytes(1, &count_byte)) {
      return false;
    }
    referred_count = count_byte[0] >> 5;
  }

  const uint32_t referred_size =
      segment_number > 65536 ? 4 : segment_number > 256 ? 2 : 1;
  const uint32_t page_assoc_size = (flags & 0x40) ? 4 : 1;
  if (!reader->Skip(referred_count * referred_size)) {
    return false;
  }
  if (!reader->Skip(page_assoc_size)) {
    return false;
  }
  if (!reader->ReadU32(data_length)) {
    return false;
  }
  *segment_type = flags & 0x3f;
  return true;
}

// Scans an interleaved (header, data, header, data, ...) JBIG2 segment
// stream for the first page-info segment (type 48) and returns its page
// width/height. Returns false if no page info is found.
bool FindPageDimensions(const std::vector<uint8_t>& stream,
                        uint32_t* width,
                        uint32_t* height) {
  ByteReader reader{stream.data(), stream.size()};
  while (reader.pos + 11 <= reader.size) {
    uint32_t segment_type = 0;
    uint32_t data_length = 0;
    if (!ParseSegmentHeader(&reader, &segment_type, &data_length)) {
      return false;
    }
    if (segment_type == 48) {
      if (!reader.ReadU32(width) || !reader.ReadU32(height)) {
        return false;
      }
      // 7.4.8.2: an unknown page height is 0xFFFFFFFF, and 7.4.8.6 then
      // requires the "page is striped" bit, whose companion field carries
      // the maximum stripe size. That is the height pdfium itself allocates
      // for such a page (jbig2_context.cpp:373), so use it here rather than
      // trying to allocate 0xFFFFFFFF rows. Skipping X/Y resolution (4 each)
      // and the flags byte lands on the 2-byte striping field.
      if (*height == 0xffffffff) {
        uint16_t striping = 0;
        if (!reader.Skip(9) || !reader.ReadU16(&striping)) {
          return false;
        }
        *height = striping & 0x7fffu;
        if (*height == 0) {
          return false;
        }
      }
      return true;
    }
    if (segment_type == 51) {
      return false;  // End-of-file segment, no page info seen.
    }
    if (data_length == 0xffffffff || !reader.Skip(data_length)) {
      return false;
    }
  }
  return false;
}

// D.4: parses the file header, if present, and returns the number of bytes
// it occupies (0 for an embedded stream, which has none) and the
// organization it declares.
bool ParseFileHeader(const std::vector<uint8_t>& file_data,
                     size_t* header_size,
                     Organization* org) {
  if (file_data.size() < 8 ||
      !std::equal(kJBIG2FileHeaderId, kJBIG2FileHeaderId + 8,
                  file_data.begin())) {
    *header_size = 0;
    *org = Organization::kEmbedded;
    return true;
  }
  if (file_data.size() < 9) {
    return false;
  }
  const uint8_t flags = file_data[8];
  *org = (flags & 0x01) ? Organization::kSequential
                        : Organization::kRandomAccess;
  // D.4.3: the page count field is present unless bit 1 says it's unknown.
  const bool page_count_known = !(flags & 0x02);
  *header_size = 9 + (page_count_known ? 4 : 0);
  return file_data.size() >= *header_size;
}

// D.2: random-access organization writes every segment header first, then
// every segment's data, in the same order -- instead of interleaving each
// segment's header with its own data, as sequential and embedded
// organization do. PDFium's decoder (CJBig2_Context::DecodeSequential)
// only understands the interleaved layout, so reorder into it here: walk
// the header block once to learn each segment's declared length, then
// re-emit (header, data) pairs by slicing the data block with those
// lengths. Unknown-length segments (7.2.7) are not reorderable this way --
// D.2 disallows them under random-access organization for exactly this
// reason, so their appearance here means the input is not spec-conformant.
bool DeinterleaveRandomAccess(const std::vector<uint8_t>& body,
                              std::vector<uint8_t>* out) {
  struct Header {
    size_t start;
    size_t end;
    uint32_t data_length;
  };
  std::vector<Header> headers;
  ByteReader reader{body.data(), body.size()};
  while (reader.pos + 11 <= reader.size) {
    const size_t start = reader.pos;
    uint32_t segment_type = 0;
    uint32_t data_length = 0;
    if (!ParseSegmentHeader(&reader, &segment_type, &data_length)) {
      return false;
    }
    if (data_length == 0xffffffff) {
      return false;
    }
    headers.push_back({start, reader.pos, data_length});
    if (segment_type == 51) {  // End-of-file: no data part, last segment.
      break;
    }
  }

  size_t data_pos = reader.pos;
  out->clear();
  out->reserve(body.size());
  for (const Header& h : headers) {
    out->insert(out->end(), body.begin() + static_cast<ptrdiff_t>(h.start),
               body.begin() + static_cast<ptrdiff_t>(h.end));
    if (data_pos + h.data_length > body.size()) {
      return false;
    }
    out->insert(out->end(),
               body.begin() + static_cast<ptrdiff_t>(data_pos),
               body.begin() + static_cast<ptrdiff_t>(data_pos + h.data_length));
    data_pos += h.data_length;
  }
  return true;
}

bool ReadFile(const std::string& path, std::vector<uint8_t>* out) {
  FILE* file = fopen(path.c_str(), "rb");
  if (!file) {
    return false;
  }
  fseek(file, 0, SEEK_END);
  const long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  if (size < 0) {
    fclose(file);
    return false;
  }
  out->resize(static_cast<size_t>(size));
  if (size > 0 && fread(out->data(), 1, static_cast<size_t>(size), file) !=
                      static_cast<size_t>(size)) {
    fclose(file);
    return false;
  }
  fclose(file);
  return true;
}

bool WritePbm(const std::string& path,
              const std::vector<uint8_t>& bitmap,
              uint32_t width,
              uint32_t height,
              uint32_t pitch) {
  FILE* file = fopen(path.c_str(), "wb");
  if (!file) {
    return false;
  }
  fprintf(file, "P4\n%u %u\n", width, height);
  const size_t row_bytes = (width + 7) / 8;
  for (uint32_t y = 0; y < height; ++y) {
    fwrite(bitmap.data() + static_cast<size_t>(y) * pitch, 1, row_bytes, file);
  }
  fclose(file);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "Usage: %s <input.jb2> [output.pbm]\n", argv[0]);
    return 2;
  }

  std::vector<uint8_t> file_data;
  if (!ReadFile(argv[1], &file_data)) {
    fprintf(stderr, "Failed to read input file '%s'\n", argv[1]);
    return 1;
  }

  if (file_data.size() < 9) {
    fprintf(stderr, "Input file is too short to be a JBIG2 stream\n");
    return 1;
  }

  size_t header_size = 0;
  Organization org;
  if (!ParseFileHeader(file_data, &header_size, &org)) {
    fprintf(stderr, "Failed to parse JBIG2 file header\n");
    return 1;
  }
  std::vector<uint8_t> body(
      file_data.begin() + static_cast<ptrdiff_t>(header_size),
      file_data.end());

  // PDFium's decoder expects an interleaved (header, data, header, data,
  // ...) segment stream, which is exactly what embedded and sequential
  // organization already are. Random-access organization is laid out
  // differently (D.2) and needs reordering into that shape first.
  std::vector<uint8_t> embedded;
  if (org == Organization::kRandomAccess) {
    if (!DeinterleaveRandomAccess(body, &embedded)) {
      fprintf(stderr, "Failed to reorder random-access JBIG2 stream\n");
      return 1;
    }
  } else {
    embedded = std::move(body);
  }

  uint32_t width = 0;
  uint32_t height = 0;
  if (!FindPageDimensions(embedded, &width, &height) || width == 0 ||
      height == 0) {
    fprintf(stderr, "Failed to locate page-info segment in JBIG2 stream\n");
    return 1;
  }

  const uint32_t pitch = ((width + 31) / 32) * 4;
  std::vector<uint8_t> bitmap(static_cast<size_t>(pitch) * height, 0);

  JBig2_DocumentContext document_context;
  Jbig2Context jbig2_context;
  FXCODEC_STATUS status = Jbig2Decoder::StartDecode(
      &jbig2_context, &document_context, width, height,
      pdfium::span(embedded), 1, {}, 0, pdfium::span(bitmap), pitch, nullptr,
      /*reject_large_regions_when_fuzzing=*/false);
  while (status == FXCODEC_STATUS::kDecodeToBeContinued) {
    status = Jbig2Decoder::ContinueDecode(&jbig2_context, nullptr);
  }
  if (status != FXCODEC_STATUS::kDecodeFinished) {
    fprintf(stderr, "JBIG2 decode failed (status %d)\n",
            static_cast<int>(status));
    return 1;
  }

  const std::string output_path =
      argc >= 3 ? argv[2] : std::string(argv[1]) + ".pbm";
  if (!WritePbm(output_path, bitmap, width, height, pitch)) {
    fprintf(stderr, "Failed to write output file '%s'\n", output_path.c_str());
    return 1;
  }

  fprintf(stderr, "Decoded %u x %u -> %s\n", width, height,
          output_path.c_str());
  return 0;
}
