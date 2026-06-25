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


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# GME-Qwen model path
GME_PATH = "Hugging-Face/gme-qwen"

# Weibo corpus dir
CORPUS_DIR = "weibo_dataset/tweets"

# Image embedding file
IMG_EMB_PKL = "weibo_dataset/img_emb_qwen25_3scale.pkl"


CHAT_SAVE_DIR = "Qwen_weibo_dataset_chat"
CAPTIONS_JSON = os.path.join(CHAT_SAVE_DIR, "weibo_captions.json")

# Output directory
MATRIX_SAVE_DIR = "Qwen_weibo_dataset_all"

# Text batch size
TEXT_BATCH_SIZE = 64

# Whether save text as vector or fake sequence
TEXT_AS_VECTOR = True
SEQ_LEN_FOR_FAKE = 150


def load_gme_model(path: str, device: str = DEVICE, dtype=DTYPE):
    model = AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else "auto"
    )
    model.eval()
    return model


def ensure_image_emb_shape(emb: np.ndarray) -> np.ndarray:
    """Ensure image embedding shape: (3,1536,1,1)"""
    arr = np.array(emb)
    if arr.ndim == 1 and len(arr) == 3:
        arr = np.stack(arr, axis=0)
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = np.squeeze(arr, axis=1)
    if arr.ndim == 4 and arr.shape == (3, 1536, 1, 1):
        return arr
    if arr.ndim == 2 and arr.shape == (3, 1536):
        arr = arr[:, :, None, None]
        return arr
    if arr.ndim == 1 and arr.shape[0] == 1536:
        arr = np.stack([arr, arr, arr], axis=0)
        arr = arr[:, :, None, None]
        return arr
    return arr


def image_key_from_any(path_or_name: str) -> str:
    """Normalize any image path or name to a base key (without extension)."""
    base = os.path.basename(path_or_name)
    base_no_ext = os.path.splitext(base)[0]
    return base_no_ext


def load_captions(json_path: str) -> Dict[str, str]:
    """Load captions into a dict keyed by image base name."""
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


def get_weibo_matrix(
        data_type: str,
        all_img_embed: Dict[str, np.ndarray],
        cap_dict: Dict[str, str]
) -> Tuple[List[str], List[np.ndarray], List[int], List[int], List[str], List[str]]:
    """
    Construct text-image-label triplets and newly added lists.
    """
    rumor_content = open(f"{CORPUS_DIR}/{data_type}_rumor.txt", "r", encoding="utf-8").readlines()
    nonrumor_content = open(f"{CORPUS_DIR}/{data_type}_nonrumor.txt", "r", encoding="utf-8").readlines()

    text_list, image_list, labels = [], [], []
    indices, image_ids, captions = [], [], []

    current_index = 0

    for idx in range(2, len(rumor_content), 3):
        text = rumor_content[idx].strip()
        if not text:
            continue

        images = rumor_content[idx - 1].split("|")
        for image in images:
            img_name = image_key_from_any(image)
            if img_name in all_img_embed:
                emb = ensure_image_emb_shape(np.array(all_img_embed[img_name]))

                text_list.append(text)
                image_list.append(emb)
                labels.append(1)
                indices.append(current_index)
                image_ids.append(img_name)
                captions.append(cap_dict.get(img_name, ""))

                current_index += 1
                break  # Only take first matched image

    for idx in range(2, len(nonrumor_content), 3):
        text = nonrumor_content[idx].strip()
        if not text:
            continue

        images = nonrumor_content[idx - 1].split("|")
        for image in images:
            img_name = image_key_from_any(image)
            if img_name in all_img_embed:
                emb = ensure_image_emb_shape(np.array(all_img_embed[img_name]))

                text_list.append(text)
                image_list.append(emb)
                labels.append(0)
                indices.append(current_index)
                image_ids.append(img_name)
                captions.append(cap_dict.get(img_name, ""))

                current_index += 1
                break

    return text_list, image_list, labels, indices, image_ids, captions



def batch_text_embeddings(
        model,
        texts: List[str],
        batch_size: int = TEXT_BATCH_SIZE
) -> np.ndarray:
    all_vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Texts"):
            batch = texts[i:i + batch_size]
            if not batch:
                continue
            feats = model.get_text_embeddings(texts=batch)
            if isinstance(feats, np.ndarray):
                vec = feats
            else:
                vec = feats.detach().cpu().numpy()

            all_vecs.append(vec)
            del feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 1536), dtype=np.float32)


def maybe_expand_to_seq(mat: np.ndarray, seq_len: int = SEQ_LEN_FOR_FAKE):
    if mat.ndim == 2:
        mat = np.repeat(mat[:, None, :], repeats=seq_len, axis=1)
    return mat



def preview_samples(texts, image_ids, captions, labels, title="样本"):
    print(f"\n===== 🔍 预览 {title} (前 3 条) =====")
    limit = min(3, len(texts))
    for i in range(limit):
        print(f"【第{i + 1}条】")
        print(f"Label: {labels[i]}")
        print(f"Image: {image_ids[i]}")
        print(f"Text: {texts[i][:150]}{'...' if len(texts[i]) > 150 else ''}")
        print(f"[CAPTION] {captions[i][:150]}{'...' if len(captions[i]) > 150 else ''}")



def main():
    os.makedirs(MATRIX_SAVE_DIR, exist_ok=True)

    print("Loading GME-Qwen...")
    model = load_gme_model(GME_PATH)

    print("Loading Image Embeddings...")
    with open(IMG_EMB_PKL, "rb") as f:
        all_img_embed = pkl.load(f)

    print(f"Loading Captions from: {CAPTIONS_JSON}")
    cap_dict = load_captions(CAPTIONS_JSON)

    print("Building Train Dataset...")
    train_texts, train_images, train_labels, train_indices, train_image_ids, train_caps = get_weibo_matrix("train",
                                                                                                           all_img_embed,
                                                                                                           cap_dict)

    print("Building Test Dataset...")
    test_texts, test_images, test_labels, test_indices, test_image_ids, test_caps = get_weibo_matrix("test",
                                                                                                     all_img_embed,
                                                                                                     cap_dict)

    preview_samples(train_texts, train_image_ids, train_caps, train_labels, title="训练集")
    preview_samples(test_texts, test_image_ids, test_caps, test_labels, title="测试集")

    # 1. Main text embeddings
    print("\nExtracting Train Text Embeddings...")
    train_text_embed = batch_text_embeddings(model, train_texts)
    print("Extracting Test Text Embeddings...")
    test_text_embed = batch_text_embeddings(model, test_texts)

    # 2. BIs/IPs embeddings processing
    train_raw_path = os.path.join(CHAT_SAVE_DIR, "train_raw.jsonl")
    test_raw_path = os.path.join(CHAT_SAVE_DIR, "test_raw.jsonl")

    def parse_jsonl(path):
        data_jsons = []
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.lower() == "assistant":
                    j = i + 1
                    json_str_lines = []
                    while j < len(lines):
                        json_line = lines[j]
                        json_str_lines.append(json_line)
                        if json_line.strip() == "}":
                            break
                        j += 1
                    try:
                        data = json.loads("\n".join(json_str_lines))
                        data_jsons.append(data)
                    except json.JSONDecodeError:
                        data_jsons.append({})
                    i = j
                i += 1
        return data_jsons

    print(f"Reading background/evidence for train: {train_raw_path}")
    train_data_jsons = parse_jsonl(train_raw_path)
    print(f"Reading background/evidence for test: {test_raw_path}")
    test_data_jsons = parse_jsonl(test_raw_path)

    def build_bi_ip_texts(indices: List[int], data_jsons: List[Dict]) -> List[str]:
        texts: List[str] = []
        for idx in indices:
            data = data_jsons[idx] if idx < len(data_jsons) else {}
            B = " ".join(data.get("BIs", []))
            I = " ".join(data.get("IPs", []))
            if B.strip() == "": B = "模型未返回有效背景信息"
            if I.strip() == "": I = "模型未返回有效依据"
            texts.append(f"[BACKGROUND] {B} [EVIDENCE] {I}")
        return texts

    train_BIsIPs_texts = build_bi_ip_texts(train_indices, train_data_jsons)
    test_BIsIPs_texts = build_bi_ip_texts(test_indices, test_data_jsons)

    print("Extracting Train BIs/IPs Embeddings...")
    train_BIsIPs_embed = batch_text_embeddings(model, train_BIsIPs_texts)
    print("Extracting Test BIs/IPs Embeddings...")
    test_BIsIPs_embed = batch_text_embeddings(model, test_BIsIPs_texts)

    # 3. Image captions -> text embeddings
    print("Extracting Train Image Caption Embeddings...")
    train_image_caption_embed = batch_text_embeddings(model, train_caps)
    print("Extracting Test Image Caption Embeddings...")
    test_image_caption_embed = batch_text_embeddings(model, test_caps)

    # Expand to Seq if configured
    if not TEXT_AS_VECTOR:
        train_text_embed = maybe_expand_to_seq(train_text_embed)
        test_text_embed = maybe_expand_to_seq(test_text_embed)
        train_BIsIPs_embed = maybe_expand_to_seq(train_BIsIPs_embed)
        test_BIsIPs_embed = maybe_expand_to_seq(test_BIsIPs_embed)
        train_image_caption_embed = maybe_expand_to_seq(train_image_caption_embed)
        test_image_caption_embed = maybe_expand_to_seq(test_image_caption_embed)

    train_image_embed = np.array(train_images)
    test_image_embed = np.array(test_images)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_labels = np.array(test_labels, dtype=np.int64)

    # Print Shapes
    print("\n----- Shapes -----")
    print("Train Text:", train_text_embed.shape)
    print("Train Image:", train_image_embed.shape)
    print("Train Label:", train_labels.shape)
    print("Train BIs/IPs:", train_BIsIPs_embed.shape)
    print("Train Caption:", train_image_caption_embed.shape)

    print("\nTest Text:", test_text_embed.shape)
    print("Test Image:", test_image_embed.shape)
    print("Test Label:", test_labels.shape)
    print("Test BIs/IPs:", test_BIsIPs_embed.shape)
    print("Test Caption:", test_image_caption_embed.shape)

    # Save
    np.save(f"{MATRIX_SAVE_DIR}/train_text_embed.npy", train_text_embed)
    np.save(f"{MATRIX_SAVE_DIR}/train_image_embed.npy", train_image_embed)
    np.save(f"{MATRIX_SAVE_DIR}/train_label.npy", train_labels)
    np.save(f"{MATRIX_SAVE_DIR}/train_text_BIs_IPs_embed.npy", train_BIsIPs_embed)
    np.save(f"{MATRIX_SAVE_DIR}/train_image_caption_embed.npy", train_image_caption_embed)

    np.save(f"{MATRIX_SAVE_DIR}/test_text_embed.npy", test_text_embed)
    np.save(f"{MATRIX_SAVE_DIR}/test_image_embed.npy", test_image_embed)
    np.save(f"{MATRIX_SAVE_DIR}/test_label.npy", test_labels)
    np.save(f"{MATRIX_SAVE_DIR}/test_text_BIs_IPs_embed.npy", test_BIsIPs_embed)
    np.save(f"{MATRIX_SAVE_DIR}/test_image_caption_embed.npy", test_image_caption_embed)

    print(f"\n[Done] All matrices saved to: {MATRIX_SAVE_DIR}")


if __name__ == "__main__":
    main()
    # cap_dict = load_captions(CAPTIONS_JSON)
    # preview_samples(cap_dict)