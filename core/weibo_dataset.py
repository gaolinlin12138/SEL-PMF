#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDatasetV2(Dataset):
    def __init__(self, text_file, image_file, label_file, biip_file, caption_file):
        # -------- text --------
        text = np.load(text_file, allow_pickle=True)
        if text.ndim == 2 and text.shape[1] == 1536:
            text = np.repeat(text[:, None, :], repeats=150, axis=1)
        assert text.ndim == 3 and text.shape[1:] == (150, 1536), f"text shape {text.shape} invalid"
        self.text = torch.from_numpy(text).float()

        # -------- image --------
        image = np.load(image_file, allow_pickle=True)
        if image.ndim == 5 and image.shape[1:] == (3, 1536, 1, 1):
            image = image.squeeze(-1).squeeze(-1)  # -> (N,3,1536)
        assert image.ndim == 3 and image.shape[1:] == (3, 1536), f"image shape {image.shape} invalid"
        self.image = torch.from_numpy(image).float()

        # -------- label --------
        label = np.load(label_file, allow_pickle=True)
        self.label = torch.from_numpy(label).long()

        # -------- biip --------
        biip = np.load(biip_file, allow_pickle=True)
        if biip.ndim == 2 and biip.shape[1] == 1536:
            biip = np.repeat(biip[:, None, :], repeats=150, axis=1)
        assert biip.ndim == 3 and biip.shape[1:] == (150, 1536), f"biip shape {biip.shape} invalid"
        self.biip = torch.from_numpy(biip).float()

        # -------- caption --------
        caption = np.load(caption_file, allow_pickle=True)
        if caption.ndim == 2 and caption.shape[1] == 1536:
            caption = np.repeat(caption[:, None, :], repeats=150, axis=1)
        assert caption.ndim == 3 and caption.shape[1:] == (150, 1536), f"caption shape {caption.shape} invalid"
        self.caption = torch.from_numpy(caption).float()

        # -------- length check --------
        n = len(self.text)
        assert len(self.image) == n and len(self.label) == n and len(self.biip) == n and len(self.caption) == n, \
            f"length mismatch: text={len(self.text)}, image={len(self.image)}, label={len(self.label)}, biip={len(self.biip)}, caption={len(self.caption)}"

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # text_seq: [150,1536], image_3sc: [3,1536], label: [], biip_seq: [150,1536], caption_seq: [150,1536]
        return self.text[idx], self.image[idx], self.label[idx], self.biip[idx], self.caption[idx]
