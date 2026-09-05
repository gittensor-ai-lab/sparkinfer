#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sparkinfer_server {

// A decoded clip: N frames, all the same size, already RGB8.
struct DecodedVideo {
    int width = 0, height = 0;
    std::vector<std::vector<unsigned char>> frames;   // each [height * width * 3]
    // ORIGINAL frame numbers that were sampled, at the container's own frame rate. Not 0..N-1:
    // the timestamps the prompt carries are computed from these over `fps`, so subsampling a
    // 30fps clip to 2fps still yields real elapsed seconds rather than frame ordinals.
    std::vector<int> frame_indices;
    double fps = 0.0;                                 // the SOURCE frame rate
};

// Decodes a video container to sampled RGB8 frames by invoking ffmpeg/ffprobe as subprocesses.
//
// Why a subprocess and not a linked decoder: this repo vendors single-header dependencies
// (stb_image.h, httplib.h) and there is no single-header H.264/VP9 decoder worth trusting.
// Linking libavcodec would pull a large LGPL dependency into a CUDA serving binary and make the
// build materially harder on every box this runs on. ffmpeg-as-a-tool keeps the build unchanged
// and the dependency inspectable, at the cost of a fork/exec per request and a runtime
// requirement that is checked and reported explicitly rather than discovered as a crash.
//
// The bytes handed to ffmpeg are UNTRUSTED. They arrive over the wire from whoever called the
// server, so this path is bounded on every axis that could be used to exhaust the box: total
// input bytes, decode wall-clock, sampled frame count, and per-frame pixel count. ffmpeg runs
// with no network protocols enabled, so a crafted container cannot turn the decoder into an
// outbound fetch primitive.
//
//   max_frames:  hard ceiling on sampled frames. Frames become prompt tokens -- roughly
//                (h/32)*(w/32) of them per temporal PAIR -- so this is a context budget, not
//                just a memory one.
//   target_fps:  sampling rate. <= 0 samples the clip's own rate.
bool decode_video(const unsigned char* bytes, size_t n, int max_frames, double target_fps,
                  DecodedVideo& out, std::string& err);

// Parses an OpenAI-style video_url value. data: URLs ONLY, for exactly the reason parse_image_url
// refuses remote images: an inference server that fetches caller-supplied URLs is an SSRF
// primitive, and that is the deployer's decision to make, not a default.
bool parse_video_url(const std::string& url, std::vector<unsigned char>& bytes, std::string& err);

// Is a usable ffmpeg (and ffprobe) actually present? Checked so the server can say "video input
// needs ffmpeg on PATH" instead of failing inside a pipe with an empty read.
bool video_decoder_available(std::string* detail = nullptr);

// Ceiling on one clip's encoded bytes, applied BEFORE the decoder is handed anything. Larger than
// the image ceiling because video legitimately is, but still bounded: the decode happens before
// any pixel budget can apply.
constexpr size_t kMaxVideoBytes = 256u * 1024u * 1024u;

// Default sampling. 2 fps over at most 32 frames is 16 temporal groups; at a typical 4:3 grid
// that is already thousands of prompt tokens, so these defaults are chosen against the context
// budget rather than against fidelity.
constexpr int    kDefaultMaxVideoFrames = 32;
constexpr double kDefaultVideoFps       = 2.0;

// Wall-clock ceiling for one decode. A malformed or adversarial container can make a decoder
// spin; without this the request thread would hang rather than error.
constexpr int    kVideoDecodeTimeoutSec = 60;

}  // namespace sparkinfer_server
