#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, re, gc
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

# ======= paths =======
CORPUS_DIR = "weibo_dataset/tweets"
SAVE_DIR = "Qwen_weibo_dataset_chat"
QWEN_PATH = "Hugging-Face/Qwen3-VL-4B-Instruct"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda:0"
DTYPE = torch.float16

NEUTRAL_PROMPT = (
    '请针对以下新闻文章[t] 提供相关背景信息，并解释新闻为真或假的客观依据，'
    '请以以下JSON格式呈现，每点50字：\n'
    '{ "BKs": ["b1"], "AEs": ["c1"] }\n新闻：[t]'
)


# NEUTRAL_PROMPT = (
#     'Please provide relevant background information for the following news article [t], and provide the objective basis for determining whether the news is true or false.'
#     'Present the response in the following JSON format, with each point around 50 words:\n'
#     '{ "BKs": ["b1"], "AEs": ["c1"] }\nNews：[t]'
# )

def build_prompt(text):
    return NEUTRAL_PROMPT.replace("[t]", text)


def extract_json_block(s):
    m = re.search(r"\{[\s\S]*\}", s)
    return m.group(0) if m else "{}"


def generate_one(tokenizer, model, text):
    prompt = build_prompt(text)

    if hasattr(tokenizer, "apply_chat_template"):
        content = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        content = prompt

    inp = tokenizer(content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=256, do_sample=False, temperature=0.0
        )
    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    j = extract_json_block(resp)
    try:
        data = json.loads(j)
    except:
        data = {"BIs": [""], "IPs": [""]}

    B = " ".join(data.get("BIs", [""]))
    I = " ".join(data.get("IPs", [""]))

    if B.strip() == "":
        B = "模型未返回有效背景信息"
    if I.strip() == "":
        I = "模型未返回有效依据"

    return B, I, j


def load_weibo(type):
    f_r = open(f"{CORPUS_DIR}/{type}_rumor.txt").read().splitlines()
    f_nr = open(f"{CORPUS_DIR}/{type}_nonrumor.txt").read().splitlines()

    texts = []
    labels = []
    # rumor: label=1
    for i in range(2, len(f_r), 3):
        texts.append(f_r[i].strip())
        labels.append(1)
    # non-rumor: label=0
    for i in range(2, len(f_nr), 3):
        texts.append(f_nr[i].strip())
        labels.append(0)

    return texts, labels


def process_split(name, tokenizer, model):
    texts, labels = load_weibo(name)
    BIs, IPs, raws = [], [], []

    for t in tqdm(texts, desc=f"Generating {name} BI/IP"):
        B, I, raw = generate_one(tokenizer, model, t)  # 完全盲测
        BIs.append(B)
        IPs.append(I)
        raws.append(raw)
        torch.cuda.empty_cache()
        gc.collect()

    with open(f"{SAVE_DIR}/{name}_BIs.txt", "w", encoding="utf-8") as f:
        for i, b in enumerate(BIs):
            f.write(f"[{i}] {b}\n\n")

    with open(f"{SAVE_DIR}/{name}_IPs.txt", "w", encoding="utf-8") as f:
        for i, ip in enumerate(IPs):
            f.write(f"[{i}] {ip}\n\n")

    with open(f"{SAVE_DIR}/{name}_raw.jsonl", "w", encoding="utf-8") as f:
        for raw in raws:
            f.write(raw + "\n")


def main():
    print("[Load Qwen3-VL-4B-Instruct]")
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH,
        trust_remote_code=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_PATH,
        trust_remote_code=True,
        device_map=DEVICE,
        dtype=torch.float16
    ).eval()

    process_split("train", tokenizer, model)
    process_split("test", tokenizer, model)


if __name__ == "__main__":
    main()
