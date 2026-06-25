#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import pickle as pkl
from typing import List, Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

# -----------------------------
# Global Configs
# -----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # adjust if needed

# GME-Qwen model path
GME_PATH = "Hugging-Face/gme-qwen"

# Twitter dataset corpus (single-line ==sep== format under tweets/)
CORPUS_DIR = "twitter_dataset/tweets"
# Image embedding pickle (1536-dim only)
IMG_EMB_PKL = "twitter_dataset/img_emb_qwen25_3scale.pkl"
# Directory for zero-shot generated BIs/IPs and captions for Twitter
CHAT_SAVE_DIR = "Qwen_twitter_dataset_chat"
# Captions JSON path (entries: {"image": "/abs/path/to/img.jpg", "caption": "..."})
CAPTIONS_JSON = os.path.join(CHAT_SAVE_DIR, "twitter_captions.json")
# Output directory for .npy matrices
MATRIX_SAVE_DIR = "Qwen_twitter_dataset_all"
# Text batch size for embedding
TEXT_BATCH_SIZE = 16
TEXT_AS_VECTOR = False
SEQ_LEN = 90


def load_gme_model(path: str, device: str = DEVICE, dtype=DTYPE):
    model = AutoModel.from_pretrained(
        path,
        device_map=device if device != "cpu" else "auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model


def ensure_image_emb_shape(emb: np.ndarray) -> np.ndarray:
    """Normalize to (3,1536,1,1); only 1536-d is supported."""
    arr = np.array(emb)
    if arr.ndim == 1 and arr.shape[0] == 1536:
        arr = np.stack([arr, arr, arr], axis=0)  # (3,1536)
    if arr.ndim == 2 and arr.shape == (3, 1536):
        arr = arr[:, :, None, None]              # (3,1536,1,1)
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = np.squeeze(arr, axis=1)            # squeeze extra batch-like dim -> (3,1536,1,1)
    return arr


def image_key_from_any(path_or_name: str) -> str:
    """Normalize any image path or name to a base key (without extension) used in pkl/captions."""
    base = os.path.basename(path_or_name)
    base_no_ext = os.path.splitext(base)[0]
    return base_no_ext


def load_captions(json_path: str) -> Dict[str, str]:
    """Load captions into a dict keyed by image base name (no extension)."""
    cap = {}
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            img = e.get("image", "")
            text = e.get("caption", "")
            key = image_key_from_any(img)
            cap[key] = text
    return cap


def split_images_field(field: str) -> List[str]:
    """Split an image field that may be 'imgA,imgB' or a single token; return list of cleaned entries."""
    if not field:
        return []
    parts = []
    # primary separator: comma (",")
    for chunk in field.split(','):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def parse_sep_line(line: str, default_label: int | None = None) -> Dict:
    parts = [p.strip() for p in line.strip().split(" ==sep== ")]
    if len(parts) >= 4:
        data_id, text, images_field, label_str = parts[0], parts[1], parts[2], parts[3]
        label = None
        try:
            label = int(label_str)
        except Exception:
            label = default_label
    elif len(parts) == 3:
        # Heuristic: if first token looks like numeric id, treat as id+text+images, label from default
        p0 = parts[0]
        if p0.isdigit():
            data_id, text, images_field = parts[0], parts[1], parts[2]
            label = default_label
        else:
            # likely: text ==sep== images ==sep== label
            data_id, text, images_field = "", parts[0], parts[1]
            try:
                label = int(parts[2])
            except Exception:
                label = default_label
    else:
        return {"id": "", "text": "", "image_first": "", "label": default_label}

    imgs = split_images_field(images_field)
    first_img = imgs[0] if imgs else ""
    return {"id": data_id, "text": text, "image_first": first_img, "label": label}


def parse_twitter_sep_file(txt_path: str, default_label: int | None) -> List[Dict]:
    if not os.path.isfile(txt_path):
        return []
    records: List[Dict] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = parse_sep_line(line, default_label)
            # First image only; allow empty if nothing parsed
            records.append(rec)
    return records


def get_twitter_lists(data_type: str, all_img_embed: Dict[str, np.ndarray], cap_dict: Dict[str, str]):
    rumor_path = f"{CORPUS_DIR}/{data_type}_rumor.txt"
    nonrumor_path = f"{CORPUS_DIR}/{data_type}_nonrumor.txt"

    rumor_recs = parse_twitter_sep_file(rumor_path, default_label=1)
    nonrumor_recs = parse_twitter_sep_file(nonrumor_path, default_label=0)

    texts: List[str] = []
    images: List[np.ndarray] = []
    labels: List[int] = []
    indices: List[int] = []
    image_ids: List[str] = []
    captions: List[str] = []

    # rumor first (label=1)
    for idx, r in enumerate(rumor_recs):
        key = image_key_from_any(r.get("image_first", ""))
        if not key:
            continue
        if key not in all_img_embed:
            continue
        img_emb = ensure_image_emb_shape(np.array(all_img_embed[key]))
        images.append(img_emb)
        texts.append(r.get("text", ""))
        labels.append(int(r.get("label", 1)))
        indices.append(idx)
        image_ids.append(key)
        captions.append(cap_dict.get(key, ""))

    base = len(rumor_recs)
    for j, r in enumerate(nonrumor_recs):
        key = image_key_from_any(r.get("image_first", ""))
        if not key:
            continue
        if key not in all_img_embed:
            continue
        img_emb = ensure_image_emb_shape(np.array(all_img_embed[key]))
        images.append(img_emb)
        texts.append(r.get("text", ""))
        labels.append(int(r.get("label", 0)))
        indices.append(base + j)
        image_ids.append(key)
        captions.append(cap_dict.get(key, ""))

    return texts, images, labels, indices, image_ids, captions


def batch_text_embeddings(model, texts: List[str], batch_size: int = TEXT_BATCH_SIZE, device: str = DEVICE) -> np.ndarray:
    all_vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts (GME-Qwen)"):
            batch = texts[i : i + batch_size]
            if not batch:
                continue
            feats = model.get_text_embeddings(texts=batch)
            vec = feats if isinstance(feats, np.ndarray) else feats.detach().cpu().numpy()
            all_vecs.append(vec)
            del feats
            torch.cuda.empty_cache()
            gc.collect()
    return np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 1536), dtype=np.float32)


def maybe_expand_to_seq(mat: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    if mat.ndim == 2 and mat.shape[-1] == 1536:
        mat = np.repeat(mat[:, None, :], repeats=seq_len, axis=1)
    return mat


def preview_samples(cap_dict: Dict[str, str]):
    print("===== 🔍 预览样本内容（前5条训练 + 前5条测试） =====")

    def show_samples(records: List[Dict], title: str):
        print(f"===== 🟢 {title}前5条样本 =====")
        for idx, r in enumerate(records[:5], start=1):
            text = r.get("text", "")
            key = image_key_from_any(r.get("image_first", "")) if r.get("image_first") else "无图片"
            caption = cap_dict.get(key, "（无图像描述）")
            label = r.get("label", "?")
            print(f"【第{idx}条】")
            print(f"Label: {label}")
            print(f"Image: {key}")
            print(f"Text: {text[:150]}{'...' if len(text) > 150 else ''}")
            print(f"[CAPTION] {caption[:200]}{'...' if len(caption) > 200 else ''}")

    train_path_r = f"{CORPUS_DIR}/train_rumor.txt"
    train_path_n = f"{CORPUS_DIR}/train_nonrumor.txt"
    test_path_r = f"{CORPUS_DIR}/test_rumor.txt"
    test_path_n = f"{CORPUS_DIR}/test_nonrumor.txt"

    train_records = parse_twitter_sep_file(train_path_r, 1) + parse_twitter_sep_file(train_path_n, 0)
    test_records = parse_twitter_sep_file(test_path_r, 1) + parse_twitter_sep_file(test_path_n, 0)

    show_samples(train_records, "训练集")
    show_samples(test_records, "测试集")

def main():
    os.makedirs(MATRIX_SAVE_DIR, exist_ok=True)

    print(f"[Info] Loading GME-Qwen model from: {GME_PATH}")
    model = load_gme_model(GME_PATH, DEVICE, DTYPE)

    print(f"[Info] Loading image embeddings from: {IMG_EMB_PKL}")
    with open(IMG_EMB_PKL, "rb") as f:
        all_img_embed = pkl.load(f)

    print(f"[Info] Loading captions from: {CAPTIONS_JSON}")
    cap_dict = load_captions(CAPTIONS_JSON)

    # Optional: quick preview before heavy embedding
    preview_samples(cap_dict)

    print("[Info] Building train sample lists (==sep==)...")
    train_texts, train_image_list, train_labels, train_indices, train_image_ids, train_caps = \
        get_twitter_lists("train", all_img_embed, cap_dict)

    print("[Info] Building test sample lists (==sep==)...")
    test_texts, test_image_list, test_labels, test_indices, test_image_ids, test_caps = \
        get_twitter_lists("test", all_img_embed, cap_dict)

    # Main text embeddings
    print("[Info] Embedding main news text for train samples...")
    train_text_embed = batch_text_embeddings(model, train_texts, TEXT_BATCH_SIZE, DEVICE)
    print("[Info] Embedding main news text for test samples...")
    test_text_embed = batch_text_embeddings(model, test_texts, TEXT_BATCH_SIZE, DEVICE)

    # BIs/IPs (background + evidence) from JSONL (optional; graceful fallback)
    train_raw_path = os.path.join(CHAT_SAVE_DIR, "train_raw.jsonl")
    test_raw_path = os.path.join(CHAT_SAVE_DIR, "test_raw.jsonl")
    print(f"[Info] Reading background/evidence from: {train_raw_path} (optional)")
    train_data_jsons = []
    if os.path.isfile(train_raw_path):
        train_data_jsons = []
        with open(train_raw_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.lower() == "assistant":
                j = i + 1
                json_str_lines: List[str] = []
                while j < len(lines):
                    json_line = lines[j]
                    json_str_lines.append(json_line)
                    if json_line.strip() == "}":
                        break
                    j += 1
                try:
                    data = json.loads("".join(json_str_lines))
                    train_data_jsons.append(data)
                except json.JSONDecodeError:
                    train_data_jsons.append({})
                i = j
            i += 1

    print(f"[Info] Reading background/evidence from: {test_raw_path} (optional)")
    test_data_jsons = []
    if os.path.isfile(test_raw_path):
        with open(test_raw_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.lower() == "assistant":
                j = i + 1
                json_str_lines: List[str] = []
                while j < len(lines):
                    json_line = lines[j]
                    json_str_lines.append(json_line)
                    if json_line.strip() == "}":
                        break
                    j += 1
                try:
                    data = json.loads("".join(json_str_lines))
                    test_data_jsons.append(data)
                except json.JSONDecodeError:
                    test_data_jsons.append({})
                i = j
            i += 1

    def build_bi_ip_texts(indices: List[int], data_jsons: List[Dict]) -> List[str]:
        texts: List[str] = []
        for idx in indices:
            data = data_jsons[idx] if idx < len(data_jsons) else {}
            B = " ".join(data.get("BIs", []))
            I = " ".join(data.get("IPs", []))
            if B.strip() == "":
                B = "模型未返回有效背景信息"
            if I.strip() == "":
                I = "模型未返回有效依据"
            texts.append(f"[BACKGROUND] {B} [EVIDENCE] {I}")
        return texts

    train_BIsIPs_texts = build_bi_ip_texts(train_indices, train_data_jsons)
    test_BIsIPs_texts = build_bi_ip_texts(test_indices, test_data_jsons)

    print("[Info] Embedding background+evidence text for train samples...")
    train_BIsIPs_embed = batch_text_embeddings(model, train_BIsIPs_texts, TEXT_BATCH_SIZE, DEVICE)
    print("[Info] Embedding background+evidence text for test samples...")
    test_BIsIPs_embed = batch_text_embeddings(model, test_BIsIPs_texts, TEXT_BATCH_SIZE, DEVICE)

    # Image captions -> text embeddings
    print("[Info] Embedding image captions for train samples...")
    train_image_caption_embed = batch_text_embeddings(model, train_caps, TEXT_BATCH_SIZE, DEVICE)
    print("[Info] Embedding image captions for test samples...")
    test_image_caption_embed = batch_text_embeddings(model, test_caps, TEXT_BATCH_SIZE, DEVICE)

    # Optional: expand vectors to sequences
    if not TEXT_AS_VECTOR:
        train_text_embed = maybe_expand_to_seq(train_text_embed, SEQ_LEN)
        test_text_embed = maybe_expand_to_seq(test_text_embed, SEQ_LEN)
        train_BIsIPs_embed = maybe_expand_to_seq(train_BIsIPs_embed, SEQ_LEN)
        test_BIsIPs_embed = maybe_expand_to_seq(test_BIsIPs_embed, SEQ_LEN)
        train_image_caption_embed = maybe_expand_to_seq(train_image_caption_embed, SEQ_LEN)
        test_image_caption_embed = maybe_expand_to_seq(test_image_caption_embed, SEQ_LEN)

    # Convert image lists and labels to numpy arrays
    train_image_mat = np.array(train_image_list)
    test_image_mat = np.array(test_image_list)
    train_labels_arr = np.array(train_labels, dtype=np.int64)
    test_labels_arr = np.array(test_labels, dtype=np.int64)

    # Shapes
    print("train_text_embed shape:", train_text_embed.shape)
    print("train_image_embed shape:", train_image_mat.shape)
    print("train_label shape:", train_labels_arr.shape)
    print("train_text_BIs_IPs_embed shape:", train_BIsIPs_embed.shape)
    print("train_image_caption_embed shape:", train_image_caption_embed.shape)
    print("test_text_embed shape:", test_text_embed.shape)
    print("test_image_embed shape:", test_image_mat.shape)
    print("test_label shape:", test_labels_arr.shape)
    print("test_text_BIs_IPs_embed shape:", test_BIsIPs_embed.shape)
    print("test_image_caption_embed shape:", test_image_caption_embed.shape)

    # Save
    np.save(os.path.join(MATRIX_SAVE_DIR, "train_text_embed"), train_text_embed)
    np.save(os.path.join(MATRIX_SAVE_DIR, "train_image_embed"), train_image_mat)
    np.save(os.path.join(MATRIX_SAVE_DIR, "train_label"), train_labels_arr)
    np.save(os.path.join(MATRIX_SAVE_DIR, "train_text_BIs_IPs_embed"), train_BIsIPs_embed)

    np.save(os.path.join(MATRIX_SAVE_DIR, "test_text_embed"), test_text_embed)
    np.save(os.path.join(MATRIX_SAVE_DIR, "test_image_embed"), test_image_mat)
    np.save(os.path.join(MATRIX_SAVE_DIR, "test_label"), test_labels_arr)
    np.save(os.path.join(MATRIX_SAVE_DIR, "test_text_BIs_IPs_embed"), test_BIsIPs_embed)

    np.save(os.path.join(MATRIX_SAVE_DIR, "train_image_caption_embed"), train_image_caption_embed)
    np.save(os.path.join(MATRIX_SAVE_DIR, "test_image_caption_embed"), test_image_caption_embed)

    print("[Done] All embedding matrices saved to:", MATRIX_SAVE_DIR)


if __name__ == "__main__":
    main()
    # cap_dict = load_captions(CAPTIONS_JSON)
    # preview_samples(cap_dict)