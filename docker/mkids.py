import json, sys
from tokenizers import Tokenizer
import os
D = os.environ.get("MODEL_DIR", "/data/final")
tok = Tokenizer.from_file(f"{D}/tokenizer.json")
prompts = {
 "chat": "Explain why the sky appears blue during the day and red at sunset.",
 "code": "Write a Python function that merges two sorted lists into one sorted list.",
 "math": "A train travels 120 km at 60 km/h, then 180 km at 90 km/h. What is the average speed?",
 "json": "Return a JSON object describing a book with title, author, year and three tags.",
}
which = sys.argv[1]
msg = [{"role":"user","content":prompts[which]}]
# render with the shipped chat template, thinking off
import re
tpl = open(f"{D}/chat_template.jinja").read()
from jinja2 import Template
t = Template(tpl)
try:
    text = t.render(messages=msg, add_generation_prompt=True, enable_thinking=False, tools=None)
except Exception as e:
    text = "<|im_start|>user\n" + prompts[which] + "<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(text, add_special_tokens=False).ids
print(" ".join(str(i) for i in ids))
