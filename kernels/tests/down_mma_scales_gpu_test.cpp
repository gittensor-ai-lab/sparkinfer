// Exercise Q4_K scale/min decoding and the down-MMA scratch lifecycle through the
// production launcher. Logical weights are generated before packing, so the
// oracle never uses the production metadata decoder. Group-constant, exactly
// representable activations make the Q8_1 scale and sum independently known.
#include "sparkinfer/kernels/moe.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Inputs {
    Buffer<__nv_bfloat16> gate, up, output;
    Buffer<float> weights, h, scratch;
    Buffer<int> ids;
    std::vector<__nv_bfloat16> host_up;
    std::vector<float> host_weights;
    Inputs(int H, int F) : gate(32u*F), up(32u*F), output(32u*H),
        weights(32), h(32u*F), scratch(32u*H), ids(32),
        host_up(32u*F), host_weights(32) {
        std::vector<__nv_bfloat16> g(32u*F, __float2bfloat16(32.f));
        check(cudaMemcpy(gate.p, g.data(), g.size()*sizeof(g[0]), cudaMemcpyHostToDevice));
        check(cudaMemset(ids.p, 0, 32*sizeof(int)));
        check(cudaMemset(scratch.p, 0, 32u*H*sizeof(float)));
    }
};

void run_shape(int H, int F, const std::vector<int>& widths, cudaStream_t* streams) {
    const int blocks = F / 256;
    const size_t bytes = (size_t)2 * H * blocks * 144;
    std::vector<unsigned char> packed(bytes, 0);
    std::vector<double> dot_coefficient(2*H, 0), min_coefficient(2*H, 0);
    for (int r = 0; r < 2*H; ++r) {
        for (int b = 0; b < blocks; ++b) {
            auto* p = packed.data() + ((size_t)r*blocks+b)*144;
            const float d = std::ldexp(1.f, -10-(r%3));
            const float dm = std::ldexp(1.f, -11-(b%2));
            const __half hd = __float2half(d), hm = __float2half(dm);
            std::memcpy(p, &hd, 2); std::memcpy(p+2, &hm, 2);
            unsigned sc[8], mn[8];
            for (int g = 0; g < 8; ++g) {
                sc[g] = (13*r+7*b+9*g)%63+1;
                mn[g] = (17*r+11*b+5*g)%64;
                int sum_q = 0;
                for (int j = 0; j < 32; ++j) {
                    const unsigned q = (3*r+5*b+7*g+j)%16;
                    p[16+(g/2)*32+j] |= (unsigned char)(q << (4*(g&1)));
                    sum_q += (int)q;
                }
                // Eight distinct power-of-two activation magnitudes distinguish
                // every scale/min group, including a permutation of two groups.
                dot_coefficient[r] += (1u<<g)*(double)d*sc[g]*sum_q;
                min_coefficient[r] += (1u<<g)*(double)dm*mn[g];
            }
            // Standard GGML Q4_K six-bit scale packing, independent of the
            // kernel's pair-at-a-time uint16 decoder.
            for (int g = 0; g < 4; ++g) {
                p[4+g] = (unsigned char)(sc[g] | ((sc[g+4] >> 4) << 6));
                p[8+g] = (unsigned char)(mn[g] | ((mn[g+4] >> 4) << 6));
                p[12+g] = (unsigned char)((sc[g+4]&15) | ((mn[g+4]&15)<<4));
            }
        }
    }
    Buffer<unsigned char> down(bytes);
    check(cudaMemcpy(down.p, packed.data(), bytes, cudaMemcpyHostToDevice));
    Inputs a(H,F), b(H,F);
    Inputs* inputs[] = {&a,&b};
    std::vector<int> second_expert(32,1);
    check(cudaMemcpy(b.ids.p,second_expert.data(),32*sizeof(int),cudaMemcpyHostToDevice));
    // Initialization used the default stream; the two nonblocking streams must
    // not race its copies, even with pageable host buffers.
    check(cudaDeviceSynchronize());
    const float values[] = {.25f,-.25f,.5f,-.5f,1.f,-1.f,.125f,-.125f};

    auto fill = [&](int lane, int M, bool zero) {
        auto& x = *inputs[lane];
        for (int row = 0; row < M; ++row) {
            const float u = zero ? 0.f : values[row%8]*(lane+1)/128.f;
            for (int col = 0; col < F; ++col)
                x.host_up[(size_t)row*F+col] = __float2bfloat16(u*(1u<<((col/32)%8)));
            x.host_weights[row] = row%3==0 ? 1.f : (row%3==1 ? .5f : -1.f);
        }
        check(cudaMemcpyAsync(x.up.p, x.host_up.data(), (size_t)M*F*sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice, streams[lane]));
        check(cudaMemcpyAsync(x.weights.p, x.host_weights.data(), M*sizeof(float),
                              cudaMemcpyHostToDevice, streams[lane]));
    };
    auto launch = [&](int lane, int M) {
        auto& x = *inputs[lane];
        sparkinfer::kernels::launch_moe_expert_ffn_q4k(
            nullptr,nullptr,nullptr,down.p,12,12,12,x.ids.p,x.weights.p,
            x.output.p,x.h.p,x.scratch.p,M,1,H,F,nullptr,streams[lane],false,x.gate.p,x.up.p);
        check(cudaPeekAtLastError());
    };
    auto verify = [&](int lane, int M, bool zero) {
        auto& x = *inputs[lane];
        check(cudaStreamSynchronize(streams[lane]));
        std::vector<__nv_bfloat16> out((size_t)M*H);
        check(cudaMemcpy(out.data(),x.output.p,out.size()*sizeof(out[0]),cudaMemcpyDeviceToHost));
        for (int row = 0; row < M; ++row) {
            // SiLU(32) rounds to32 in float; Q8 codes are exactly +/-127.
            // The per-group factors were folded into the oracle coefficients.
            // Scaling these normal half values by powers of two is exact.
            const float hv = zero ? 0.f : 32.f*values[row%8]*(lane+1)/128.f;
            const double qd = __half2float(__float2half(std::fabs(hv)/127.f));
            const double qs = __half2float(__float2half(hv*32.f));
            const int qi = hv>0 ? 127 : (hv<0 ? -127 : 0);
            for (int col = 0; col < H; ++col) {
                const double want = (qd*qi*dot_coefficient[lane*H+col]-qs*min_coefficient[lane*H+col])
                                    * x.host_weights[row];
                const double got = __bfloat162float(out[(size_t)row*H+col]);
                // BF16 rounding plus FP32 split-K summation, not a relaxed
                // quantization comparison. Zero must remain exactly zero.
                const double tolerance = zero ? 0.0 : std::max(.015625, std::fabs(want)*.008);
                if (!std::isfinite(got) || std::fabs(got-want)>tolerance) {
                    std::printf("FAIL H=%d F=%d M=%d stream=%d row=%d col=%d got=%.9g want=%.9g tol=%.9g\n",
                                H,F,M,lane,row,col,got,want,tolerance);
                    throw std::runtime_error("down MMA numerical or scratch-lifecycle mismatch");
                }
            }
        }
    };
    for (int M : widths) {
        for (int lane = 0; lane < 2; ++lane) fill(lane,M,false);
        for (int lane = 0; lane < 2; ++lane) launch(lane,M);
        for (int lane = 0; lane < 2; ++lane) verify(lane,M,false);
        cudaGraph_t graphs[2]{};
        cudaGraphExec_t execs[2]{};
        for (int lane = 0; lane < 2; ++lane) {
            check(cudaStreamBeginCapture(streams[lane],cudaStreamCaptureModeGlobal));
            launch(lane,M);
            check(cudaStreamEndCapture(streams[lane],&graphs[lane]));
            check(cudaGraphInstantiate(&execs[lane],graphs[lane],nullptr,nullptr,0));
        }
        // Reuse the same captured launches across A -> zero -> A. Two streams
        // carry distinct inputs; four replays expose missed accumulator resets.
        for (bool zero : {false,true,false}) {
            for (int lane = 0; lane < 2; ++lane) fill(lane,M,zero);
            for (int lane = 0; lane < 2; ++lane) {
                for (int repeat = 0; repeat < 4; ++repeat)
                    check(cudaGraphLaunch(execs[lane],streams[lane]));
            }
            for (int lane = 0; lane < 2; ++lane) verify(lane,M,zero);
        }
        for (int lane = 0; lane < 2; ++lane) {
            check(cudaGraphExecDestroy(execs[lane]));
            check(cudaGraphDestroy(graphs[lane]));
        }
        std::printf("PASS down MMA scales H=%d F=%d M=%d eager + two-stream graph A/zero/A\n",H,F,M);
    }
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) return 77;
    setenv("SPARKINFER_DOWN_MMA","1",1);
    setenv("SPARKINFER_DOWN_MMA_MINROWS","8",1);
    setenv("SPARKINFER_DOWN_MMVQ","1",1);
    setenv("SPARKINFER_DOWN_Q4K","1",1);
    setenv("SPARKINFER_DOWN_SPLITK_S_Q4","8",1);
    setenv("SPARKINFER_MMA_ASTAGE","1",1);
    setenv("SPARKINFER_DOWN_SPLITK_WIDE_ROWS","5",1);
    setenv("SPARKINFER_DOWN_SPLITK_WIDE_S","2",1);
    cudaStream_t streams[2]{};
    try {
        check(cudaStreamCreateWithFlags(&streams[0],cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&streams[1],cudaStreamNonBlocking));
        run_shape(256,256,{8,9,16,17,32},streams);    // one K split
        run_shape(1024,2304,{8,16,32},streams);       // uneven multiple splits
        run_shape(6656,19968,{8,9,16,17,32},streams);// real Muse widths
        check(cudaStreamDestroy(streams[0]));
        check(cudaStreamDestroy(streams[1]));
    } catch (const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what());
        return 1;
    }
    return 0;
}
