#include "image_input.hpp"

#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_NO_STDIO          // requests arrive as bytes; no filesystem path is ever taken
#define STBI_NO_FAILURE_STRINGS_MUTABLE
#include "stb_image.h"

namespace sparkinfer_server {

bool decode_image(const unsigned char* bytes, size_t n, DecodedImage& out, std::string& err) {
    if (!bytes || n == 0) { err = "empty image payload"; return false; }
    if (n > kMaxImageBytes) { err = "image payload exceeds the size limit"; return false; }
    int w = 0, h = 0, ch = 0;
    // 3 = force RGB. stb handles the source's own channel count and any palette.
    unsigned char* p = stbi_load_from_memory(bytes, (int)n, &w, &h, &ch, 3);
    if (!p) { err = std::string("image decode failed: ") + (stbi_failure_reason() ?: "unknown"); return false; }
    if (w <= 0 || h <= 0) { stbi_image_free(p); err = "decoded image has zero extent"; return false; }
    out.width = w; out.height = h;
    out.rgb.assign(p, p + (size_t)w * h * 3);
    stbi_image_free(p);
    return true;
}

bool base64_decode(const std::string& in, std::vector<unsigned char>& out, std::string& err) {
    static signed char T[256];
    static bool init = false;
    if (!init) {
        std::memset(T, -1, sizeof(T));
        const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) T[(unsigned char)A[i]] = (signed char)i;
        init = true;
    }
    out.clear();
    out.reserve(in.size() / 4 * 3 + 3);
    int acc = 0, bits = 0;
    size_t pad = 0;
    for (char c : in) {
        if (c == '\n' || c == '\r') continue;   // wrapped base64 is common and harmless
        if (c == '=') { pad++; continue; }
        if (pad) { err = "base64: data after padding"; return false; }
        const signed char v = T[(unsigned char)c];
        if (v < 0) { err = "base64: invalid character"; return false; }
        acc = (acc << 6) | v; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((unsigned char)((acc >> bits) & 0xFF)); }
    }
    if (pad > 2) { err = "base64: too much padding"; return false; }
    return true;
}

bool parse_image_url(const std::string& url, std::vector<unsigned char>& bytes, std::string& err) {
    if (url.rfind("data:", 0) != 0) {
        err = "only data: URLs are accepted; remote image fetching is disabled by policy "
              "(it would let a request drive server-side HTTP to arbitrary hosts)";
        return false;
    }
    const size_t comma = url.find(',');
    if (comma == std::string::npos) { err = "malformed data: URL (no comma)"; return false; }
    const std::string meta = url.substr(5, comma - 5);
    if (meta.find("base64") == std::string::npos) {
        err = "data: URL must be base64-encoded";
        return false;
    }
    // Roughly bound the ENCODED length too -- 4/3 expansion means decoding a huge string
    // allocates before any size check inside decode_image could fire.
    if (url.size() - comma - 1 > kMaxImageBytes / 3 * 4 + 4) {
        err = "image payload exceeds the size limit";
        return false;
    }
    return base64_decode(url.substr(comma + 1), bytes, err);
}

}  // namespace sparkinfer_server
