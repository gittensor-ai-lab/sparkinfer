#include "video_input.hpp"
#include "image_input.hpp"      // base64_decode -- data: URL handling is shared with images

#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace sparkinfer_server {

namespace {

// Runs argv with `input` on its stdin and collects stdout. Returns false on spawn failure, a
// nonzero exit, a timeout, or when stdout exceeds cap.
//
// Written with fork/exec rather than popen because popen goes through /bin/sh, which would mean
// building a command STRING out of values that are ultimately request-influenced. execvp takes an
// argv array, so nothing is ever parsed by a shell and there is no quoting to get wrong.
bool run_tool(const std::vector<std::string>& argv,
              const unsigned char* input, size_t input_n,
              std::string& out, size_t cap, int timeout_sec, std::string& err) {
    int in_pipe[2] = {-1, -1}, out_pipe[2] = {-1, -1};
    if (pipe(in_pipe) != 0) { err = "pipe: " + std::string(std::strerror(errno)); return false; }
    if (pipe(out_pipe) != 0) {
        close(in_pipe[0]); close(in_pipe[1]);
        err = "pipe: " + std::string(std::strerror(errno)); return false;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]); close(in_pipe[1]); close(out_pipe[0]); close(out_pipe[1]);
        err = "fork: " + std::string(std::strerror(errno));
        return false;
    }
    if (pid == 0) {
        // Child. Wire the pipes, silence stderr, exec. Anything that fails here _exits rather
        // than returning, so a failed exec can never fall back into the parent's code path.
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        close(in_pipe[0]); close(in_pipe[1]); close(out_pipe[0]); close(out_pipe[1]);
        std::vector<char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
        cargv.push_back(nullptr);
        execvp(cargv[0], cargv.data());
        _exit(127);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);
    // Don't die on SIGPIPE when the child exits before consuming all input -- ffmpeg legitimately
    // stops reading once it has the frames it was asked for.
    signal(SIGPIPE, SIG_IGN);

    // Feed stdin and drain stdout in one loop. Writing everything first would deadlock as soon as
    // the child's stdout pipe fills, which for rawvideo is immediate.
    fcntl(in_pipe[1], F_SETFL, O_NONBLOCK);
    fcntl(out_pipe[0], F_SETFL, O_NONBLOCK);
    size_t written = 0;
    bool over_cap = false, timed_out = false;
    out.clear();
    const time_t deadline = time(nullptr) + timeout_sec;
    char buf[1 << 16];
    bool stdin_open = true;
    while (true) {
        if (time(nullptr) > deadline) { timed_out = true; break; }
        if (stdin_open && written < input_n) {
            const ssize_t w = write(in_pipe[1], input + written, input_n - written);
            if (w > 0) written += (size_t)w;
            else if (w < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                close(in_pipe[1]); stdin_open = false;   // child closed its stdin; not an error
            }
        } else if (stdin_open) {
            close(in_pipe[1]); stdin_open = false;
        }
        const ssize_t r = read(out_pipe[0], buf, sizeof(buf));
        if (r > 0) {
            if (out.size() + (size_t)r > cap) { over_cap = true; break; }
            out.append(buf, (size_t)r);
        } else if (r == 0) {
            break;                                        // child closed stdout: done
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
            break;
        } else if (!stdin_open) {
            struct timespec ts = {0, 2 * 1000 * 1000};    // 2ms; nothing to write, nothing to read
            nanosleep(&ts, nullptr);
        }
    }
    if (stdin_open) close(in_pipe[1]);
    close(out_pipe[0]);

    if (timed_out || over_cap) kill(pid, SIGKILL);
    int status = 0;
    waitpid(pid, &status, 0);

    if (timed_out) { err = "video decode timed out"; return false; }
    if (over_cap)  { err = "video decode produced more data than the frame budget allows"; return false; }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
            err = argv[0] + " not found on PATH";
        } else {
            err = argv[0] + " failed (exit " +
                  std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + ")";
        }
        return false;
    }
    return true;
}

// ffmpeg is given untrusted bytes, so every protocol it could reach out with is turned off. A
// container can otherwise reference external URLs (concat/hls playlists), which would make the
// decoder an SSRF primitive -- the exact thing parse_video_url refuses to be.
void append_hardening(std::vector<std::string>& a) {
    // `pipe` ONLY. A container can otherwise reference external URLs (concat/hls playlists),
    // which would turn the decoder into an outbound fetch primitive -- exactly what
    // parse_video_url refuses to be. Note this also excludes `file`, so a crafted playlist cannot
    // read local paths either, which is why the payload is piped rather than written to a temp
    // file even though a temp file would be simpler for seek-heavy containers.
    a.push_back("-protocol_whitelist"); a.push_back("pipe");
    // NO -nostdin here. It looks like sensible hardening and is actively wrong for this call
    // shape: stdin IS the input (-i pipe:0), and -nostdin makes ffmpeg/ffprobe refuse to read it,
    // failing with a bare nonzero exit that reads like a corrupt container.
}

bool parse_double(const std::string& s, double& v) {
    // ffprobe reports frame rates as "30000/1001" as often as "25".
    const size_t slash = s.find('/');
    try {
        if (slash != std::string::npos) {
            const double num = std::stod(s.substr(0, slash)), den = std::stod(s.substr(slash + 1));
            if (den == 0.0) return false;
            v = num / den;
        } else {
            v = std::stod(s);
        }
    } catch (...) { return false; }
    return true;
}

}  // namespace

bool video_decoder_available(std::string* detail) {
    std::string out, err;
    if (!run_tool({"ffmpeg", "-version"}, nullptr, 0, out, 1 << 20, 10, err)) {
        if (detail) *detail = "ffmpeg unavailable: " + err;
        return false;
    }
    if (!run_tool({"ffprobe", "-version"}, nullptr, 0, out, 1 << 20, 10, err)) {
        if (detail) *detail = "ffprobe unavailable: " + err;
        return false;
    }
    return true;
}

bool parse_video_url(const std::string& url, std::vector<unsigned char>& bytes, std::string& err) {
    if (url.rfind("data:", 0) != 0) {
        err = "video_url must be a data: URL; this server does not fetch remote video URLs "
              "(fetching caller-supplied URLs is an SSRF primitive and is a deployment decision, "
              "not a default)";
        return false;
    }
    const size_t comma = url.find(',');
    if (comma == std::string::npos) { err = "data: URL has no comma separator"; return false; }
    const std::string meta = url.substr(5, comma - 5);
    if (meta.find("base64") == std::string::npos) {
        err = "data: URL must be base64-encoded";
        return false;
    }
    // Bound the ENCODED length before decoding: base64 expands ~4/3, so checking after would
    // already have allocated the payload this limit exists to refuse.
    if (url.size() - comma - 1 > (kMaxVideoBytes / 3) * 4 + 4) {
        err = "video payload exceeds the size limit";
        return false;
    }
    if (!base64_decode(url.substr(comma + 1), bytes, err)) return false;
    if (bytes.size() > kMaxVideoBytes) { err = "video payload exceeds the size limit"; return false; }
    return true;
}

bool decode_video(const unsigned char* bytes, size_t n, int max_frames, double target_fps,
                  DecodedVideo& out, std::string& err) {
    if (!bytes || n == 0) { err = "empty video payload"; return false; }
    if (n > kMaxVideoBytes) { err = "video payload exceeds the size limit"; return false; }
    if (max_frames <= 0) max_frames = kDefaultMaxVideoFrames;

    // --- probe: native size and frame rate, before decoding a single frame -------------------
    std::string probe;
    {
        std::vector<std::string> a = {"ffprobe", "-v", "error"};
        append_hardening(a);
        a.insert(a.end(), {"-select_streams", "v:0",
                           "-show_entries", "stream=width,height,r_frame_rate",
                           "-of", "default=noprint_wrappers=1:nokey=0",
                           "-i", "pipe:0"});
        if (!run_tool(a, bytes, n, probe, 1 << 16, kVideoDecodeTimeoutSec, err)) {
            err = "video probe failed: " + err;
            return false;
        }
    }
    int W = 0, H = 0; double src_fps = 0.0;
    {
        auto field = [&](const char* key) -> std::string {
            const std::string k = std::string(key) + "=";
            const size_t p = probe.find(k);
            if (p == std::string::npos) return "";
            const size_t e = probe.find('\n', p);
            return probe.substr(p + k.size(), (e == std::string::npos ? probe.size() : e) - p - k.size());
        };
        W = std::atoi(field("width").c_str());
        H = std::atoi(field("height").c_str());
        if (!parse_double(field("r_frame_rate"), src_fps)) src_fps = 0.0;
    }
    if (W <= 0 || H <= 0) { err = "video has no decodable video stream"; return false; }
    // A single frame's pixels are bounded here, before ffmpeg is asked for any. Without this a
    // legitimate 8K clip would allocate ~100MB per frame and the frame ceiling alone would not
    // save the box.
    if ((long)W * H > 8192L * 8192L) { err = "video frame dimensions are too large"; return false; }

    const double fps = target_fps > 0.0 ? target_fps : (src_fps > 0.0 ? src_fps : kDefaultVideoFps);

    // --- decode: rawvideo rgb24 at the sampling rate, capped at max_frames -------------------
    const size_t frame_bytes = (size_t)W * H * 3;
    std::string raw;
    {
        std::vector<std::string> a = {"ffmpeg", "-v", "error"};
        append_hardening(a);
        a.insert(a.end(), {"-i", "pipe:0",
                           "-vf", "fps=" + std::to_string(fps),
                           "-frames:v", std::to_string(max_frames),
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"});
        // +1 frame of slack so a decoder that emits one extra is caught by the frame loop below
        // rather than tripping the byte cap and being reported as a budget error.
        if (!run_tool(a, bytes, n, raw, frame_bytes * (size_t)(max_frames + 1),
                      kVideoDecodeTimeoutSec, err)) {
            err = "video decode failed: " + err;
            return false;
        }
    }
    const size_t n_frames = raw.size() / frame_bytes;
    if (n_frames == 0) { err = "video decoded to zero frames"; return false; }

    out.width = W;
    out.height = H;
    out.fps = src_fps > 0.0 ? src_fps : fps;
    out.frames.clear();
    out.frame_indices.clear();
    out.frames.reserve(n_frames);
    out.frame_indices.reserve(n_frames);
    for (size_t i = 0; i < n_frames && i < (size_t)max_frames; i++) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(raw.data()) + i * frame_bytes;
        out.frames.emplace_back(p, p + frame_bytes);
        // Map sampled index back onto the SOURCE timeline, so timestamps read as real elapsed
        // seconds. ffmpeg's fps filter emits frame i at t = i/fps, which is source frame
        // round(i * src_fps / fps).
        const double step = (out.fps > 0.0 && fps > 0.0) ? out.fps / fps : 1.0;
        out.frame_indices.push_back((int)(i * step + 0.5));
    }
    return true;
}

}  // namespace sparkinfer_server
