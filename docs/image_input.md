# Image input

sparkinfer accepts images as **input** on checkpoints that ship a vision tower. It does not
generate images.

The tower loads automatically when the checkpoint's `config.json` has a `vision_config` block —
Qwen3.8-27B does. A text-only checkpoint simply has none, `has_vision()` stays false, and a
request carrying an image is refused rather than answered from its text alone (a silently dropped
image produces a fluent, confident description of something the model never saw).

## API

Standard OpenAI content parts on `POST /v1/chat/completions`:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}},
      {"type": "text", "text": "What object is shown in this image?"}
    ]
  }]
}
```

PNG, JPEG and BMP decode — the only three stb_image decoders compiled in. Alpha is dropped rather
than composited, matching what the reference processor does when it converts to RGB.

Several images in one request work, and they may be different sizes; each expands to the token
count its own resized grid needs. Text and image parts interleave in the order given.

### `data:` URLs only

A remote `http(s)` URL is **refused**, not fetched:

```
image 0: only data: URLs are accepted; remote image fetching is disabled by policy
(it would let a request drive server-side HTTP to arbitrary hosts)
```

An inference server that fetches URLs on request is an SSRF primitive — internal metadata
endpoints, private hosts, anything the box can reach. Whether to accept that is a deployment
decision for whoever runs the server, not a default.

### Other endpoints

`/v1/tokenize` and `/v1/score` reject requests containing images. Neither can answer honestly: an
image's token count depends on its resized grid, so tokenize would report a number short by
hundreds of tokens, and a scoring endpoint returning logprobs for the text alone defeats the one
thing it exists for. Non-image non-text parts (audio) are still ignored everywhere, so mixed
payloads from provider-agnostic clients keep working.

Images in a `system`/`developer` message are rejected, mirroring the chat template, which raises
on exactly that.

## Token accounting

The template emits **one** `<|image_pad|>` per image. The processor — not the template — expands
it, because only the processor knows the resized grid:

    tokens = (grid_h / merge) * (grid_w / merge)      # merge = spatial_merge_size = 2

A 1024x1536 image is a 96x64 patch grid, so 1536 tokens. Budget for that: images dominate the
prompt. Expansion happens before the context-length check, so an image that does not fit is
rejected up front rather than failing inside prefill.

## Verification

The tower and the preprocessing are both diffed against **transformers itself**, not against a
reimplementation:

```bash
# tower: our CUDA vs transformers' Qwen3_5VisionModel, on identical pixels
build/runtime/vision_preprocess_check  $MODEL_DIR image.png --out /tmp/px.bin
build/runtime/vision_forward_check     $MODEL_DIR --pixels /tmp/px.bin --gh 28 --gw 28 --out /tmp/ours.bin
python bench/scripts/vision_hf_check.py $MODEL_DIR --pixels /tmp/px.bin --gh 28 --gw 28 \
                                        --compare /tmp/ours.bin --image image.png
```

That needs `torch` + `transformers` (CPU is fine — the reference forward does not need the GPU).
The `--image` flag additionally runs the real `AutoImageProcessor` and diffs our preprocessing
against it.

Both gates judge against a **measured** floor rather than a constant: the tower against the bf16
noise floor computed per invocation (it moves with grid size — 0.99887 at 784 patches, 0.99562 at
2204), and preprocessing against one uint8 LSB, which is the irreducible disagreement from PIL
accumulating its 8-bit passes in fixed point.

`bench/scripts/vision_ref.py` is a hand-written NumPy reference kept for fast iteration. It is
**not** the gate: it and the CUDA once shared an omission — neither applied the tower's rotary
position embeddings — and agreed with each other at cosine 0.99915 while both disagreed with
transformers at 0.77. A reference written by the same hand as the code can only catch mistakes its
author did not also make.

## Not supported

Video input, and checkpoints with a non-empty `deepstack_visual_indexes` (refused at load rather
than silently ignored). The tower runs in bf16.
