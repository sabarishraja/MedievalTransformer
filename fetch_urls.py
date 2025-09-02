import re, requests
from pathlib import Path

CANTERBURY_URL = "https://www.gutenberg.org/cache/epub/22120/pg22120.txt"
EARNEST_URL    = "https://www.gutenberg.org/cache/epub/844/pg844.txt"

def strip_gutenberg(txt: str) -> str:
    s = re.search(r"\*\*\* START OF(.*)\*\*\*", txt)
    e = re.search(r"\*\*\* END OF(.*)\*\*\*", txt)
    if s and e and s.end() < e.start():
        txt = txt[s.end():e.start()]
    return txt.replace("\r\n", "\n").strip()

def clean_canterbury(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"_(?:[^_]{0,40})_", "", text)  # italic/underscore apparatus
    text = re.sub(r"^\s*\d{1,5}\.?\s*$", "", text, flags=re.MULTILINE)           # bare line numbers
    text = re.sub(r"^\s*[A-Z]\.\s*\d{1,5}\.?.*$", "", text, flags=re.MULTILINE)  # "B. 1270. ..."
    text = re.sub(r"^\s*\d{1,5}\.\s*[A-Z]\..*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\(\s*\d{1,5}\s*\)", "", text)
    text = re.sub(r"(?<![\w'])\d{1,5}(?![\w'])", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

print("[data] downloading Canterbury (cleaning)…")
raw_c = requests.get(CANTERBURY_URL, timeout=60).text
core_c = clean_canterbury(strip_gutenberg(raw_c))
Path("input.txt").write_text(core_c, encoding="utf-8")
print("[data] wrote input.txt (Canterbury) with", len(core_c), "chars")

print("[data] downloading The Importance of Being Earnest (no cleaning)…")
raw_e = requests.get(EARNEST_URL, timeout=60).text
core_e = strip_gutenberg(raw_e)
Path("finetune_input.txt").write_text(core_e, encoding="utf-8")
print("[data] wrote finetune_input.txt (Earnest) with", len(core_e), "chars")
