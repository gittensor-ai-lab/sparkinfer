#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sparkinfer_server {

// Decoded image, always 3-channel RGB8 regardless of what the source encoded.
struct DecodedImage {
    int width = 0, height = 0;
    std::vector<unsigned char> rgb;   // [height * width * 3]
};

// Decodes PNG / JPEG / BMP bytes to RGB8. Those three and no others: image_input.cpp sets
// STBI_ONLY_PNG/JPEG/BMP, so stb's GIF, TGA, PSD, HDR, PIC and PNM decoders are not compiled in
// -- fewer parsers reachable from untrusted request bytes.
//
// Alpha is DROPPED, not composited: stb is asked for 3 channels, so an RGBA source loses its
// alpha rather than being flattened onto a background. The vision tower has no alpha channel and
// the reference processor converts to RGB the same way, so this matches -- but it does mean a
// transparent PNG renders as whatever RGB its transparent pixels happen to carry.
bool decode_image(const unsigned char* bytes, size_t n, DecodedImage& out, std::string& err);

// Decodes standard base64 (RFC 4648, with '=' padding). Rejects any character outside the
// alphabet rather than skipping it: a data: URL that fails to decode cleanly is far more likely
// to be a malformed request than something worth salvaging, and silently dropping bytes would
// hand the decoder a truncated image.
bool base64_decode(const std::string& in, std::vector<unsigned char>& out, std::string& err);

// Parses an OpenAI image_url value. Supports data: URLs only --
//     data:image/png;base64,iVBORw0KG...
// A remote http(s) URL is REFUSED rather than fetched: having an inference server fetch
// arbitrary URLs on request is an SSRF primitive (internal metadata endpoints, private hosts),
// and that is a deliberate deployment decision for whoever runs it, not a default. The error
// says so explicitly so callers know it is policy, not a parse failure.
bool parse_image_url(const std::string& url, std::vector<unsigned char>& bytes, std::string& err);

// Byte ceiling for one decoded data: URL, before decoding. Guards against a request that expands
// into gigabytes of pixels; the pixel budget in smart_resize bounds what reaches the tower, but
// the DECODE itself happens first and has to be bounded separately.
constexpr size_t kMaxImageBytes = 32u * 1024u * 1024u;

}  // namespace sparkinfer_server
