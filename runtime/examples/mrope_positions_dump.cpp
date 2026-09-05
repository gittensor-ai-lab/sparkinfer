// Dumps runtime MRoPE position ids as JSON for bench/scripts/mrope_ref_check.py to diff against
// transformers' own get_rope_index. Builds prompts that exercise the cases that actually differ
// from 1D RoPE: an image, a video's per-frame spans, both interleaved, and text on either side.
#include "sparkinfer/models/qwen_vision_preprocess.h"
#include "sparkinfer/models/qwen_vision_config.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {

struct Span { std::string kind; int t, h, w; };

// Builds a prompt with the given vision spans and the grids that describe them.
struct Case {
    std::string name;
    std::vector<int> ids;
    std::vector<Span> spans;              // for the reference harness
    std::vector<QwenVisionSpanGrid> grids;   // for our own call, same order
};

void add_run(std::vector<int>& ids, int tok, int n) { for (int i = 0; i < n; i++) ids.push_back(tok); }

}  // namespace

int main(int argc, char** argv) {
    const char* out_dir = argc > 1 ? argv[1] : ".";
    QwenVisionConfig cfg;
    const int IMG = cfg.image_token_id, VID = cfg.video_token_id;
    const int VS = cfg.vision_start_token_id, VE = cfg.vision_end_token_id;
    const int merge = cfg.spatial_merge;

    std::vector<Case> cases;

    // 1. one image between text. 4x6 pre-merge grid -> 2x3 merged -> 6 placeholder tokens.
    {
        Case c; c.name = "image";
        c.ids = {100, 101, VS};
        add_run(c.ids, IMG, (4 / merge) * (6 / merge));
        c.ids.push_back(VE); c.ids.push_back(102); c.ids.push_back(103);
        c.spans = {{"image", 1, 4, 6}};
        c.grids = {{1, 4, 6}};
        cases.push_back(c);
    }
    // 2. a video: TWO temporal groups, each its own span with a timestamp marker between them.
    //    This is the layout qwen_vision_expand_video_placeholders produces.
    {
        Case c; c.name = "video_2groups";
        c.ids = {100, VS};
        for (int g = 0; g < 2; g++) {
            c.ids.push_back(900 + g);            // "<t seconds>" stand-in
            c.ids.push_back(VS);
            add_run(c.ids, VID, (4 / merge) * (4 / merge));
            c.ids.push_back(VE);
        }
        c.ids.push_back(VE); c.ids.push_back(101);
        // Each GROUP is its own (1,h,w) entry -- the reference splits video_grid_thw by t and
        // sets t=1, so a 2-group clip is two rows, not one row with t=2.
        c.spans = {{"video", 1, 4, 4}, {"video", 1, 4, 4}};
        c.grids = {{1, 4, 4}, {1, 4, 4}};
        cases.push_back(c);
    }
    // 3. image and video interleaved -- the ordering case reexpand_vision exists to get right.
    {
        Case c; c.name = "image_then_video";
        c.ids = {100, VS};
        add_run(c.ids, IMG, (2 / merge) * (4 / merge));
        c.ids.push_back(VE);
        c.ids.push_back(101);
        c.ids.push_back(VS); c.ids.push_back(900); c.ids.push_back(VS);
        add_run(c.ids, VID, (6 / merge) * (2 / merge));
        c.ids.push_back(VE); c.ids.push_back(VE);
        c.ids.push_back(102);
        c.spans = {{"image", 1, 2, 4}, {"video", 1, 6, 2}};
        c.grids = {{1, 2, 4}, {1, 6, 2}};
        cases.push_back(c);
    }
    // 4. non-square grid where h != w, so the max(h,w) advance is observable.
    {
        Case c; c.name = "tall_image";
        c.ids = {100};
        c.ids.push_back(VS);
        add_run(c.ids, IMG, (12 / merge) * (2 / merge));
        c.ids.push_back(VE);
        for (int i = 0; i < 5; i++) c.ids.push_back(200 + i);   // text AFTER, where the advance shows
        c.spans = {{"image", 1, 12, 2}};
        c.grids = {{1, 12, 2}};
        cases.push_back(c);
    }
    // 5. text only -- must be identical to plain 1D RoPE, the property the whole design rests on.
    {
        Case c; c.name = "text_only";
        for (int i = 0; i < 12; i++) c.ids.push_back(300 + i);
        cases.push_back(c);
    }

    int rc = 0;
    for (const Case& c : cases) {
        std::vector<int> pos;
        std::string err;
        if (!qwen_vision_mrope_positions(c.ids, IMG, VID, c.grids, merge, 0, pos, err)) {
            std::fprintf(stderr, "%s: %s\n", c.name.c_str(), err.c_str());
            rc = 1;
            continue;
        }
        const std::string path = std::string(out_dir) + "/mrope_" + c.name + ".json";
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); rc = 1; continue; }
        std::fprintf(f, "{\n  \"name\": \"%s\",\n  \"image_token_id\": %d,\n  \"video_token_id\": %d,\n",
                     c.name.c_str(), IMG, VID);
        std::fprintf(f, "  \"spatial_merge\": %d,\n  \"start_pos\": 0,\n", merge);
        std::fprintf(f, "  \"token_ids\": [");
        for (size_t i = 0; i < c.ids.size(); i++) std::fprintf(f, "%s%d", i ? "," : "", c.ids[i]);
        std::fprintf(f, "],\n  \"spans\": [");
        for (size_t i = 0; i < c.spans.size(); i++)
            std::fprintf(f, "%s{\"kind\":\"%s\",\"t\":%d,\"h\":%d,\"w\":%d}", i ? "," : "",
                         c.spans[i].kind.c_str(), c.spans[i].t, c.spans[i].h, c.spans[i].w);
        std::fprintf(f, "],\n  \"positions\": [");
        for (size_t i = 0; i < pos.size(); i++) std::fprintf(f, "%s%d", i ? "," : "", pos[i]);
        std::fprintf(f, "]\n}\n");
        std::fclose(f);
        std::printf("wrote %s  (%zu tokens)\n", path.c_str(), c.ids.size());
    }
    return rc;
}
