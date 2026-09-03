// base64 + data: URL + decode behaviour, including the refusals.
//
// The refusals are the security-relevant half: a remote URL must NOT be fetched (SSRF), oversized
// payloads must be rejected before allocation, and malformed base64 must fail rather than silently
// yielding truncated bytes that stb then misinterprets.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "image_input.hpp"

using namespace sparkinfer_server;
static int g_fail = 0;
static void check(bool ok, const char* what) {
    printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
}

int main() {
    std::string err;
    std::vector<unsigned char> out;

    check(base64_decode("", out, err) && out.empty(), "empty base64 -> empty output");
    check(base64_decode("TWFu", out, err) && out.size() == 3 && !memcmp(out.data(), "Man", 3),
          "base64 'TWFu' -> 'Man'");
    check(base64_decode("TWE=", out, err) && out.size() == 2 && !memcmp(out.data(), "Ma", 2),
          "base64 with one pad char");
    check(base64_decode("TQ==", out, err) && out.size() == 1 && out[0] == 'M',
          "base64 with two pad chars");
    check(base64_decode("TWFu\nTWFu", out, err) && out.size() == 6,
          "newlines inside base64 are tolerated");
    check(!base64_decode("TW*u", out, err), "invalid base64 character is rejected");
    check(!base64_decode("TWE=Zg", out, err), "data after base64 padding is rejected");

    // A 1x1 red PNG, base64 -- the smallest real decode path.
    const std::string png1x1 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    std::vector<unsigned char> bytes;
    check(parse_image_url("data:image/png;base64," + png1x1, bytes, err) && !bytes.empty(),
          "data: URL with base64 PNG parses");
    DecodedImage img;
    check(decode_image(bytes.data(), bytes.size(), img, err) && img.width == 1 && img.height == 1
              && img.rgb.size() == 3,
          "1x1 PNG decodes to exactly 3 RGB bytes");

    check(!parse_image_url("https://example.com/cat.png", bytes, err),
          "remote https URL is REFUSED, not fetched (SSRF)");
    check(err.find("policy") != std::string::npos,
          "  ...and the error says it is policy, not a parse failure");
    check(!parse_image_url("http://169.254.169.254/latest/meta-data/", bytes, err),
          "cloud metadata URL is refused too");
    check(!parse_image_url("data:image/png,notbase64", bytes, err),
          "non-base64 data: URL is refused");
    check(!parse_image_url("data:image/png;base64", bytes, err),
          "data: URL with no comma is refused");
    check(!parse_image_url("data:image/png;base64," + std::string(kMaxImageBytes / 3 * 4 + 64, 'A'),
                           bytes, err),
          "oversized data: URL is refused before decoding");

    const unsigned char junk[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    check(!decode_image(junk, sizeof(junk), img, err), "non-image bytes fail to decode");
    check(!decode_image(nullptr, 0, img, err), "empty payload fails to decode");

    printf(g_fail ? "[FAIL] %d check(s) failed\n" : "[OK] image input handling is correct\n", g_fail);
    return g_fail ? 1 : 0;
}
