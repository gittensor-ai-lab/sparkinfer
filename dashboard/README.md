# sparkinfer dashboard

Static, self-contained status page — current **frontier**, the **optimization journey**, **vs
llama.cpp**, **emission weights**, the **auto-eval labels**, and **evaluated PRs**. Styled in the
org's template identity (purple `#7B5DFF` / lime `#D4FF12`, Sora + Work Sans). No build step, no
framework — just `index.html` + `data.js`.

## View
Open `dashboard/index.html` in a browser (it loads `dashboard/data.js`).

## Update the data
Canonical data is **`data.json`**; **`data.js`** is generated from it
(`window.SPARKINFER = <data.json>`) so the page can load it on `file://` and Pages. Edit
`data.json`, then regenerate `data.js`:
```bash
python3 -c "import json;d=json.load(open('dashboard/data.json'));open('dashboard/data.js','w').write('window.SPARKINFER = '+json.dumps(d,indent=2)+';\n')"
```

**The eval bot does this automatically.** After each evaluated PR, `eval/pr_eval_bot.py` upserts the
verdict into `prs[]` (`{num,title,areas,label,tps,delta_pct,url}`), **ratchets**
`status.frontier_tps`, regenerates `data.js`, and pushes — so the live page updates with every PR,
no manual step.

## Deploy (GitHub Pages)
It's plain static files. Enable Pages (Settings → Pages → deploy from `main`, root) and it serves at
`https://gittensor-ai-lab.github.io/sparkinfer/dashboard/`.
