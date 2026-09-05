// Video decoding — exercises the real ffmpeg subprocess path against real containers.
//
// Skips (passes) when ffmpeg is absent rather than failing: video decoding is an optional runtime
// dependency, and a box without it should not turn the whole suite red. What it must NOT do is
// pass silently while ffmpeg IS present and broken, so the skip is loud.
#include "video_input.hpp"
#include "image_input.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer_server;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("  FAIL: %s\n", what.c_str()); failures++; }
    else       std::printf("  ok:   %s\n", what.c_str());
}

// Builds a real container with ffmpeg's own synthetic source, so the test does not carry a binary
// fixture and does not depend on any file outside the build.
static bool make_test_video(const std::string& path, int w, int h, int fps, int secs) {
    std::string cmd = "ffmpeg -v error -y -f lavfi -i testsrc=size=" + std::to_string(w) + "x" +
                      std::to_string(h) + ":rate=" + std::to_string(fps) +
                      " -t " + std::to_string(secs) + " -pix_fmt yuv420p " + path + " 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

static std::vector<unsigned char> slurp(const std::string& path) {
    std::vector<unsigned char> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t)(n > 0 ? n : 0));
    if (!out.empty() && std::fread(out.data(), 1, out.size(), f) != out.size()) out.clear();
    std::fclose(f);
    return out;
}

int main() {
    std::string detail;
    if (!video_decoder_available(&detail)) {
        std::printf("SKIP: %s\n", detail.c_str());
        std::printf("(video input is an optional runtime dependency; install ffmpeg to run this)\n");
        return 0;
    }
    std::printf("ffmpeg present\n");

    const std::string path = "/tmp/sparkinfer_video_test.mp4";
    if (!make_test_video(path, 128, 96, 10, 2)) {
        std::printf("FAIL: could not synthesize a test clip\n");
        return 1;
    }
    auto bytes = slurp(path);
    check(!bytes.empty(), "test clip has bytes");

    std::string err;

    // --- 1. decode at a sampling rate below the source rate ------------------------------------
    {
        DecodedVideo v;
        const bool ok = decode_video(bytes.data(), bytes.size(), 32, 2.0, v, err);
        check(ok, "decode at 2fps succeeds" + (ok ? "" : " -- " + err));
        if (ok) {
            check(v.width == 128 && v.height == 96, "native dimensions preserved (no forced scale)");
            // 2 seconds at 2fps -> about 4 frames; encoders vary by one at the boundary.
            check(v.frames.size() >= 3 && v.frames.size() <= 5,
                  "2s @ 2fps yields ~4 frames, got " + std::to_string(v.frames.size()));
            bool sized = true;
            for (const auto& f : v.frames) sized = sized && f.size() == (size_t)128 * 96 * 3;
            check(sized, "every frame is exactly W*H*3 RGB8");
            check(v.frame_indices.size() == v.frames.size(), "one source index per frame");
            bool monotonic = true;
            for (size_t i = 1; i < v.frame_indices.size(); i++)
                monotonic = monotonic && v.frame_indices[i] > v.frame_indices[i - 1];
            check(monotonic, "source indices strictly increase");
            check(v.fps > 9.0 && v.fps < 11.0, "source fps reported as ~10, got " + std::to_string(v.fps));
            // testsrc is a colour pattern; an all-black decode would mean the pixel format or
            // the raw stride is wrong, which no size check would catch.
            bool nonblack = false;
            for (unsigned char c : v.frames[0]) if (c > 16) { nonblack = true; break; }
            check(nonblack, "decoded pixels are not uniformly black");
        }
    }

    // --- 2. the frame ceiling is a real ceiling ------------------------------------------------
    {
        DecodedVideo v;
        const bool ok = decode_video(bytes.data(), bytes.size(), 3, 10.0, v, err);
        check(ok, "decode with max_frames=3 succeeds" + (ok ? "" : " -- " + err));
        check(ok && v.frames.size() <= 3, "max_frames caps the sampled frames");
    }

    // --- 3. malformed input is refused, not crashed on -----------------------------------------
    {
        DecodedVideo v;
        std::vector<unsigned char> junk(4096, 0xA5);
        check(!decode_video(junk.data(), junk.size(), 8, 2.0, v, err), "garbage bytes rejected");
        check(!decode_video(nullptr, 0, 8, 2.0, v, err), "empty payload rejected");
        // Truncating a real container mid-stream is the realistic corruption case.
        std::vector<unsigned char> trunc(bytes.begin(), bytes.begin() + bytes.size() / 8);
        DecodedVideo v2;
        const bool t = decode_video(trunc.data(), trunc.size(), 8, 2.0, v2, err);
        check(!t || !v2.frames.empty(), "truncated clip either fails or yields real frames");
    }

    // --- 4. data: URL policy -------------------------------------------------------------------
    {
        std::vector<unsigned char> got;
        check(!parse_video_url("https://example.com/a.mp4", got, err),
              "remote http(s) video URL refused");
        check(err.find("SSRF") != std::string::npos || err.find("does not fetch") != std::string::npos,
              "refusal explains it is policy, not a parse failure");
        check(!parse_video_url("data:video/mp4,notbase64", got, err), "non-base64 data: URL refused");
        check(!parse_video_url("data:video/mp4;base64", got, err), "data: URL with no comma refused");

        // Round-trip a small real clip through a data: URL.
        static const char* b64 =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string enc;
        for (size_t i = 0; i < bytes.size(); i += 3) {
            const unsigned a = bytes[i];
            const unsigned b = i + 1 < bytes.size() ? bytes[i + 1] : 0;
            const unsigned c = i + 2 < bytes.size() ? bytes[i + 2] : 0;
            const unsigned t = (a << 16) | (b << 8) | c;
            enc += b64[(t >> 18) & 63]; enc += b64[(t >> 12) & 63];
            enc += (i + 1 < bytes.size()) ? b64[(t >> 6) & 63] : '=';
            enc += (i + 2 < bytes.size()) ? b64[t & 63] : '=';
        }
        std::vector<unsigned char> back;
        const bool ok = parse_video_url("data:video/mp4;base64," + enc, back, err);
        check(ok, "data: URL round-trips" + (ok ? "" : " -- " + err));
        check(ok && back == bytes, "decoded bytes match the original clip");
    }

    std::remove(path.c_str());
    std::printf(failures ? "\nFAILED (%d)\n" : "\nPASSED\n", failures);
    return failures ? 1 : 0;
}
