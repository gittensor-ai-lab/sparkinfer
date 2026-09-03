// End-to-end image understanding: a real image file in, generated tokens out.
//
// This is the test every other vision check cannot do. The numeric checks prove the tower
// reproduces a reference and the resize matches upstream; none of them prove the MODEL
// understands what it is being shown. A convention error that survived all of them would surface
// here as a fluent caption about the wrong thing.
//
// Usage: vision_e2e <checkpoint_dir> <image_file> <max_new> <id0> <id1> ...
//   The token ids are the chat-templated prompt containing exactly ONE image_token_id, which this
//   binary expands to the count the resized grid needs (as the reference processor does).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/safetensors.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/qwen_vision.h"
#include "sparkinfer/models/qwen_vision_preprocess.h"
#include "qwen_checkpoint.h"
#include "sparkinfer/models/qwen_vision_hf_config.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_FAILURE_STRINGS_MUTABLE
#include "stb_image.h"

using namespace sparkinfer;

int main(int argc, char** argv) {
    if (argc < 5) { printf("usage: %s <ckpt_dir> <image> <max_new> <id0> ...\n", argv[0]); return 2; }
    const std::string dir = argv[1], img_path = argv[2];
    const int max_new = atoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; i++) prompt.push_back(atoi(argv[i]));

    // ---- vision config + image ----
    QwenVisionConfig vc; std::string err;
    if (!qwen_vision_config_from_hf_json(dir, vc, err) || !vc.present) {
        printf("[FAIL] vision config: %s\n", err.c_str()); return 1;
    }
    int iw = 0, ih = 0, ch = 0;
    unsigned char* px = stbi_load(img_path.c_str(), &iw, &ih, &ch, 3);
    if (!px) { printf("[FAIL] decode %s\n", img_path.c_str()); return 1; }
    printf("image: %s  %dx%d (source channels %d)\n", img_path.c_str(), iw, ih, ch);

    std::vector<float> pixels; int gh = 0, gw = 0;
    if (!qwen_vision_preprocess(px, ih, iw, vc, pixels, &gh, &gw, err)) {
        printf("[FAIL] preprocess: %s\n", err.c_str()); stbi_image_free(px); return 1;
    }
    stbi_image_free(px);
    const int n_img = qwen_vision_num_tokens(gh, gw, vc);
    printf("resized to patch grid %dx%d (%d patches) -> %d image tokens\n", gh, gw, gh * gw, n_img);

    // ---- expand the template's single placeholder ----
    std::vector<int> ids;
    if (!qwen_vision_expand_placeholders(prompt, vc.image_token_id, {n_img}, ids, err)) {
        printf("[FAIL] expand: %s\n", err.c_str()); return 1;
    }
    printf("prompt: %zu tokens -> %zu after expansion\n", prompt.size(), ids.size());

    // ---- model ----
    GGUF g; Qwen35Config cfg; QwenCheckpointKind kind{}; std::string cerr_msg;
    if (!qwen_checkpoint_open(dir, cfg, g, kind, cerr_msg)) { printf("[FAIL] %s\n", cerr_msg.c_str()); return 1; }
    cfg.max_seq = std::max(4096, (int)ids.size() + max_new + 64);
    auto rt = Runtime::create({}); rt->initialize();
    KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
    { const char* e = getenv("SPARKINFER_KV_INT8"); kvc.int8_kv = e ? (e[0] != '0') : false; }
    kvc.layer_slot = hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);
    moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = moe::MoEEngine::create(mc);
    Qwen35Model model(cfg, &kv, engine.get());
    printf("loading checkpoint (%s) ...\n", qwen_checkpoint_kind_label(kind));
    if (!qwen_checkpoint_load(model, dir, kind)) { printf("[FAIL] checkpoint load\n"); return 1; }

    // ---- vision tower ----
    SafeTensorsModel st;
    if (!st.open(dir)) { printf("[FAIL] safetensors open\n"); return 1; }
    QwenVisionWeights vw;
    if (!load_qwen_vision_weights(st, vc, vw, err)) { printf("[FAIL] vision weights: %s\n", err.c_str()); return 1; }
    std::vector<float> emb((size_t)n_img * vc.out_hidden);
    printf("running vision tower (%d blocks) ...\n", vc.depth);
    if (!qwen_vision_forward(vw, vc, pixels.data(), gh, gw, emb.data(), err)) {
        printf("[FAIL] vision forward: %s\n", err.c_str()); free_qwen_vision_weights(vw); return 1;
    }
    free_qwen_vision_weights(vw);
    double amax = 0; for (float v : emb) if (std::fabs((double)v) > amax) amax = std::fabs((double)v);
    printf("vision embeddings: [%d, %d] absmax %.3f\n", n_img, vc.out_hidden, amax);

    // ---- stage + generate ----
    std::vector<int> pos;
    for (size_t t = 0; t < ids.size(); t++) if (ids[t] == vc.image_token_id) pos.push_back((int)t);
    if (!model.set_pending_vision(emb.data(), pos.data(), n_img, cfg.hidden)) {
        printf("[FAIL] set_pending_vision\n"); return 1;
    }
    printf("generating %d tokens ...\n", max_new);
    auto out = model.generate(ids, max_new, nullptr);
    printf("OUTPUT_IDS");
    for (int t : out) printf(" %d", t);
    printf("\n");
    return 0;
}
