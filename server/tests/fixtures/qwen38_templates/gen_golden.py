#!/usr/bin/env python3
"""Golden prompts for the Qwen3.8 chat-template test (chat_template_golden_test.cpp).

Renders each request below through the checkpoints' own Jinja templates, exactly as vLLM and
transformers do (add_generation_prompt=True), and writes the expected prompt per variant:

  pinned.jinja    gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090's template (the one
                  apply_qwen36_tools_template has mirrored since #1094)
  official.jinja  Qwen3.8-27B's own template, which Swift-Qwen3.8-27B ships unmodified

Tool calling is not covered: the server renders a tool's JSON with sorted keys and HTML/ASCII
escaping, where transformers' tojson keeps the client's key order and escapes neither -- a
divergence shared by every model, tracked separately from this test.

Regenerate after changing a case or a template:  python3 gen_golden.py > golden.json
"""
import json
import os

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))

U = lambda c: {"role": "user", "content": c}
A = lambda c, r=None: dict({"role": "assistant", "content": c}, **({"reasoning_content": r} if r is not None else {}))
S = lambda c: {"role": "system", "content": c}

# (name, messages, extra request fields)
CASES = [
    ("single", [U("Hi there")], {}),
    ("system", [S("You are helpful."), U("Hi")], {}),
    ("multi_turn", [U("Q1"), A("A1"), U("Q2")], {}),
    ("multi_turn_system", [S("Be brief."), U("Q1"), A("A1"), U("Q2")], {}),
    ("reasoning_content", [U("Q1"), A("A1", "step one\nstep two"), U("Q2")], {}),
    ("embedded_think", [U("Q1"), A("<think>\nreasoning here\n</think>\n\nA1"), U("Q2")], {}),
    ("no_preserve", [U("Q1"), A("A1", "old reasoning"), U("Q2")],
     {"chat_template_kwargs": {"preserve_thinking": False}}),
    ("two_turns_reasoning", [U("Q1"), A("A1", "r1"), U("Q2"), A("A2", "r2"), U("Q3")], {}),
    ("effort_xhigh", [U("Hi")], {"reasoning_effort": "xhigh"}),
    ("effort_medium", [U("Hi")], {"reasoning_effort": "medium"}),
    ("effort_low", [U("Hi")], {"reasoning_effort": "low"}),
    ("effort_low_system", [S("You are terse."), U("Q1"), A("A1", "r"), U("Q2")], {"reasoning_effort": "low"}),
    ("whitespace", [S("  padded system  \n"), U("  Q with spaces \n"), A("  A1 \n"), U("Q2")], {}),
]


def render(template, messages, enable_thinking, extra):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"], trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    kwargs = dict(extra.get("chat_template_kwargs", {}))
    if "reasoning_effort" in extra:
        kwargs["reasoning_effort"] = extra["reasoning_effort"]
    return env.from_string(template).render(messages=messages, add_generation_prompt=True,
                                            enable_thinking=enable_thinking, **kwargs)


def main():
    templates = {v: open(os.path.join(HERE, v + ".jinja")).read() for v in ("pinned", "official")}
    out = []
    for name, messages, extra in CASES:
        for thinking in (True, False):
            body = dict({"messages": messages, "enable_thinking": thinking}, **extra)
            out.append({"name": f"{name}/{'think' if thinking else 'nothink'}", "body": body,
                        "enable_thinking": thinking,
                        "expected": {v: render(t, messages, thinking, extra) for v, t in templates.items()}})
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
