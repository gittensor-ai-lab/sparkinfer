#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/kernels/compressed_tensors.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

int main() {
    using namespace sparkinfer::kernels;
    constexpr int M=128, N=128, K=128;
    if (!prefill_nvfp4_supported(M,N,K)) return 77;
    auto check_ct_sfb = [](int n, int k) {
        void *src=nullptr, *dst=nullptr;
        const size_t src_bytes=(size_t)n*k/16;
        const size_t dst_bytes=prefill_nvfp4_scale_bytes_b(n,k);
        bool ok=cudaMalloc(&src,src_bytes)==cudaSuccess &&
                cudaMalloc(&dst,dst_bytes)==cudaSuccess &&
                cudaMemset(src,0,src_bytes)==cudaSuccess &&
                launch_ct_nvfp4_pack_sfb(src,dst,n,k) &&
                cudaDeviceSynchronize()==cudaSuccess;
        std::printf("ct_nvfp4_pack_sfb n=%d k=%d: %s\n",n,k,ok?"OK":"FAIL");
        if(src) cudaFree(src); if(dst) cudaFree(dst);
        return ok;
    };
    // Qwen3.8-27B dense FFN gate/up and down-projection shapes. The checkpoint loader runs this
    // exact row-major UE4M3 -> CUTLASS SFB conversion before any prefill, so cover production
    // dimensions here instead of relying only on the tiny GEMM below.
    bool ok=check_ct_sfb(17408,5120) && check_ct_sfb(5120,17408);
    auto check_large_row_dequant = []() {
        constexpr int rows=65536, cols=16;
        void *packed=nullptr, *scale=nullptr, *out=nullptr;
        const size_t packed_bytes=(size_t)rows*cols/2;
        const size_t scale_bytes=(size_t)rows*cols/16;
        const size_t out_bytes=(size_t)rows*cols*2;
        bool pass=cudaMalloc(&packed,packed_bytes)==cudaSuccess &&
                  cudaMalloc(&scale,scale_bytes)==cudaSuccess &&
                  cudaMalloc(&out,out_bytes)==cudaSuccess &&
                  cudaMemset(packed,0,packed_bytes)==cudaSuccess &&
                  cudaMemset(scale,0,scale_bytes)==cudaSuccess &&
                  cudaMemset(out,0xff,out_bytes)==cudaSuccess;
        if(pass) {
            cudaGetLastError();
            launch_ct_dequant_nvfp4(packed,scale,1.f,out,rows,cols);
            pass=cudaPeekAtLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess;
            uint16_t first=0xffff, last=0xffff;
            if(pass) pass=cudaMemcpy(&first,out,sizeof(first),cudaMemcpyDeviceToHost)==cudaSuccess &&
                          cudaMemcpy(&last,static_cast<char*>(out)+out_bytes-sizeof(last),
                                     sizeof(last),cudaMemcpyDeviceToHost)==cudaSuccess &&
                          first==0 && last==0;
        }
        std::printf("ct_nvfp4_dequant rows=%d (> grid.y limit): %s\n",rows,pass?"OK":"FAIL");
        if(packed) cudaFree(packed); if(scale) cudaFree(scale); if(out) cudaFree(out);
        return pass;
    };
    ok &= check_large_row_dequant();
    std::vector<__nv_bfloat16> hA(M*K, __float2bfloat16(1.f));
    std::vector<__nv_bfloat16> hB(N*K, __float2bfloat16(1.f)), hD(M*N);
    void *A0=nullptr,*B0=nullptr,*A=nullptr,*B=nullptr,*SA=nullptr,*SB=nullptr,*D=nullptr,*W=nullptr;
    cudaMalloc(&A0,hA.size()*2); cudaMalloc(&B0,hB.size()*2); cudaMalloc(&D,hD.size()*2);
    cudaMalloc(&A,prefill_nvfp4_data_bytes(M,K)); cudaMalloc(&B,prefill_nvfp4_data_bytes(N,K));
    cudaMalloc(&SA,prefill_nvfp4_scale_bytes_a(M,K)); cudaMalloc(&SB,prefill_nvfp4_scale_bytes_b(N,K));
    size_t ws=prefill_nvfp4_workspace_bytes(M,N,K); if(ws) cudaMalloc(&W,ws);
    cudaMemcpy(A0,hA.data(),hA.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(B0,hB.data(),hB.size()*2,cudaMemcpyHostToDevice);
    ok &= launch_prefill_nvfp4_quant_a(A0,A,SA,M,K) &&
            launch_prefill_nvfp4_quant_b(B0,B,SB,N,K) &&
            launch_prefill_nvfp4_gemm(A,SA,B,SB,D,M,N,K,W) &&
            cudaDeviceSynchronize()==cudaSuccess;
    cudaMemcpy(hD.data(),D,hD.size()*2,cudaMemcpyDeviceToHost);
    float lo=1e9f,hi=-1e9f;
    for(auto v:hD){ float x=__bfloat162float(v); lo=fminf(lo,x); hi=fmaxf(hi,x); ok &= std::isfinite(x) && x>115.f && x<141.f; }
    std::printf("prefill_nvfp4_gpu_test: %s range=[%.3f,%.3f]\n",ok?"OK":"FAIL",lo,hi);
    cudaFree(A0);cudaFree(B0);cudaFree(A);cudaFree(B);cudaFree(SA);cudaFree(SB);cudaFree(D);if(W)cudaFree(W);
    return ok?0:1;
}
