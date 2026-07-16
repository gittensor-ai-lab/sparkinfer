# sparkinfer dashboard

Static frontier status page for **SP⚡RKINFER · Powered by SN74** — optimization journey, per-context benchmarks, auto-eval labels, and evaluated PRs. Styled in the org identity (purple `#9D7CFF` / lime `#D4FF12`). No build step — `index.html` + `data.js`.

**Live:** [gittensor-ai-lab.github.io/sparkinfer/dashboard/](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/)

Companion surfaces:

| Surface | What |
|---|---|
| **This dashboard** | Frontier charts, PR eval history, optimization journey |
| **[sparkinfer-web](https://github.com/gittensor-ai-lab/sparkinfer-web)** | Landing page + SparkInfer Chat UI |
| **[sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer)** | Runtime, kernels, bench, SN74 eval loop |

## What it shows

- **Target GPUs** — RTX Spark, DGX Spark, RTX 5090, RTX PRO 6000 (`sm_120` / `sm_121`)
- **SOTA spotlight** — Qwen3.6-35B-A3B frontier stats at the top
- **Optimization journey** — tok/s per landed kernel optimization
- **vs llama.cpp** — same GGUF, per-context decode on RTX 5090
- **Evaluated PRs** — bot labels, never auto-merges

Current frontier (v0.4.1, Qwen3.6 SOTA): **473 tok/s @ 128 ctx · +71%** vs llama.cpp.

## View locally

Open `dashboard/index.html` in a browser (loads `dashboard/data.js`).

## Update the data

Canonical data is **`data.json`**; **`data.js`** is generated from it
(`window.SPARKINFER = <data.json>`) so the page works on `file://` and GitHub Pages.

```bash
python3 -c "import json;d=json.load(open('dashboard/data.json'));open('dashboard/data.js','w').write('window.SPARKINFER = '+json.dumps(d,indent=2)+';\n')"
```

**The eval bot does this automatically.** After each evaluated PR, `eval/pr_eval_bot.py` upserts the
verdict into `prs[]`, ratchets `status.frontier_tps`, regenerates `data.js`, and pushes.

## Deploy (GitHub Pages)

Enable Pages (Settings → Pages → deploy from `main`, root). Serves at
`https://gittensor-ai-lab.github.io/sparkinfer/dashboard/`.
