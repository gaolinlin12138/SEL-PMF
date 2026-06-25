#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import math
import json
import copy
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from twitter_dataset import FeatureDatasetV2
from twitter_model import TriBranchMSCAN


DEVICE = "cuda:0"
NUM_WORKER = 0
BATCH_SIZE = 32
LR = 5e-4
L2 = 1e-5
NUM_EPOCH = 50
WARMUP_EPOCHS = 5
MAX_GRAD_NORM = 1.0

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

SAVE_DIR = f"TWITTER/training_results_{TIMESTAMP}"
MODEL_SAVE_DIR = os.path.join(SAVE_DIR, "models")
RESULT_SAVE_DIR = os.path.join(SAVE_DIR, "results")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)


def save_checkpoint(epoch, model, optim_task_detection, best_acc, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_task_detection_state_dict': optim_task_detection.state_dict(),
        'best_acc': best_acc,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    filename_with_timestamp = f"{filename.split('.')[0]}_{TIMESTAMP}.pth"
    torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, filename_with_timestamp))
    print(f"检查点已保存: {filename_with_timestamp}")


def save_training_history(history, filename):
    filename_with_timestamp = f"{filename.split('.')[0]}_{TIMESTAMP}.json"
    with open(os.path.join(RESULT_SAVE_DIR, filename_with_timestamp), 'w') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"训练历史已保存: {filename_with_timestamp}")


def save_config():
    config = {
        'DEVICE': DEVICE,
        'NUM_WORKER': NUM_WORKER,
        'BATCH_SIZE': BATCH_SIZE,
        'LR': LR,
        'L2': L2,
        'NUM_EPOCH': NUM_EPOCH,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'MAX_GRAD_NORM': MAX_GRAD_NORM,
        'timestamp': TIMESTAMP,
        'experiment': 'Twitter Full Model (Ours)'
    }
    with open(os.path.join(RESULT_SAVE_DIR, f'training_config_{TIMESTAMP}.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def plot_roc_pr_curves(y_true, y_probs, filename):
    plt.figure(figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(RESULT_SAVE_DIR, f"{filename}_{TIMESTAMP}.png")
    plt.savefig(save_path, dpi=300)
    print(f"ROC/PR 曲线已保存: {save_path}")
    plt.close()


def plot_tsne(features, labels, filename):
    print("正在计算 T-SNE (数据量较大时可能需要几分钟)...")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_tsne = tsne.fit_transform(features)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    # 0: Non-rumor (Blue), 1: Rumor (Red)
    plt.scatter(X_norm[labels == 0, 0], X_norm[labels == 0, 1],
                c='blue', label='Non-rumor', alpha=0.6, s=10)
    plt.scatter(X_norm[labels == 1, 0], X_norm[labels == 1, 1],
                c='red', label='Rumor', alpha=0.6, s=10)

    plt.legend()
    plt.title(f'T-SNE Visualization ({filename})')
    plt.xticks([])
    plt.yticks([])

    save_path = os.path.join(RESULT_SAVE_DIR, f"{filename}_{TIMESTAMP}.png")
    plt.savefig(save_path, dpi=300)
    print(f"T-SNE 图已保存: {save_path}")
    plt.close()


def plot_training_curves(history, lr_list, filename):
    plt.figure(figsize=(15, 10))
    keys = [('train_loss', 'test_loss', 'Loss'),
            ('train_acc', 'test_acc', 'Accuracy'),
            ('f1_rumor', 'f1_non_rumor', 'F1 Score'),
            ('precision_rumor', 'recall_rumor', 'Rumor Metrics'),
            ('precision_non_rumor', 'recall_non_rumor', 'Non-Rumor Metrics')]

    for i, (k1, k2, title) in enumerate(keys):
        plt.subplot(2, 3, i + 1)
        plt.plot(history[k1], label=k1)
        plt.plot(history[k2], label=k2)
        plt.title(title)
        plt.legend()
        plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(lr_list, label="LR", color="orange")
    plt.title("Learning Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    filename_with_timestamp = f"{filename.split('.')[0]}_{TIMESTAMP}.png"
    plt.savefig(os.path.join(RESULT_SAVE_DIR, filename_with_timestamp), dpi=300)
    plt.close()


def get_scheduler(optimizer, num_epochs, warmup_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def test(model, test_loader, return_features=False):
    model.eval()
    device = torch.device(DEVICE)
    loss_func = torch.nn.CrossEntropyLoss()

    count = 0
    loss_total = 0.0
    y_all = []
    yhat_all = []
    probs_all = []
    features_all = []

    for text_seq, image_3sc, label, biip_seq, cap_seq in tqdm(test_loader, desc="Testing"):
        text_seq = text_seq.to(device)
        image_3sc = image_3sc.to(device)
        label = label.to(device)
        biip_seq = biip_seq.to(device)
        cap_seq = cap_seq.to(device)

        if return_features:
            output, feats = model(text_seq, image_3sc, biip_seq, cap_seq, return_features=True)
            features_all.append(feats.detach().cpu().numpy())
        else:
            output = model(text_seq, image_3sc, biip_seq, cap_seq)

        loss = loss_func(output, label)
        probs = torch.softmax(output, dim=1)[:, 1]
        pred = output.argmax(1)

        loss_total += loss.item() * label.size(0)
        count += label.size(0)

        yhat_all.append(pred.detach().cpu().numpy())
        y_all.append(label.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())

    loss_test = loss_total / count
    yhat_all = np.concatenate(yhat_all, 0)
    y_all = np.concatenate(y_all, 0)
    probs_all = np.concatenate(probs_all, 0)

    final_feats = None
    if return_features:
        final_feats = np.concatenate(features_all, 0)

    acc = accuracy_score(y_all, yhat_all)
    f1_1 = f1_score(y_all, yhat_all, pos_label=1)
    r1 = recall_score(y_all, yhat_all, pos_label=1)
    p1 = precision_score(y_all, yhat_all, pos_label=1)
    f1_0 = f1_score(y_all, yhat_all, pos_label=0)
    r0 = recall_score(y_all, yhat_all, pos_label=0)
    p0 = precision_score(y_all, yhat_all, pos_label=0)

    return acc, loss_test, p1, r1, f1_1, p0, r0, f1_0, probs_all, y_all, final_feats

def train():
    print(f"本次训练时间戳: {TIMESTAMP}")
    print(f"结果保存路径: {SAVE_DIR}")
    save_config()

    device = torch.device(DEVICE)
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH

    # === Twitter 数据集路径 ===
    dataset_dir = 'Qwen_twitter_dataset_all'
    dataset_prefix = 'twitter_'

    train_text = f"{dataset_dir}/{dataset_prefix}train_text_embed.npy"
    train_image = f"{dataset_dir}/{dataset_prefix}train_image_embed.npy"
    train_label = f"{dataset_dir}/{dataset_prefix}train_label.npy"
    train_biip = f"{dataset_dir}/{dataset_prefix}train_text_BIs_IPs_embed.npy"
    train_cap = f"{dataset_dir}/{dataset_prefix}train_image_caption_embed.npy"

    test_text = f"{dataset_dir}/{dataset_prefix}test_text_embed.npy"
    test_image = f"{dataset_dir}/{dataset_prefix}test_image_embed.npy"
    test_label = f"{dataset_dir}/{dataset_prefix}test_label.npy"
    test_biip = f"{dataset_dir}/{dataset_prefix}test_text_BIs_IPs_embed.npy"
    test_cap = f"{dataset_dir}/{dataset_prefix}test_image_caption_embed.npy"

    train_set = FeatureDatasetV2(train_text, train_image, train_label, train_biip, train_cap)
    test_set = FeatureDatasetV2(test_text, test_image, test_label, test_biip, test_cap)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=NUM_WORKER, shuffle=True, drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=NUM_WORKER, shuffle=False, drop_last=False,
                             pin_memory=True)

    model = TriBranchMSCAN(embed_dim=1536, num_heads=8, dropout=0.3, out_dim=2).to(device)

    loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optim_task_detection = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = get_scheduler(optim_task_detection, num_epoch, WARMUP_EPOCHS)

    training_history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'precision_rumor': [], 'recall_rumor': [], 'f1_rumor': [],
        'precision_non_rumor': [], 'recall_non_rumor': [], 'f1_non_rumor': []
    }

    best_acc = 0.0
    lr_list = []

    for epoch in range(num_epoch):
        model.train()
        corrects = 0
        loss_total = 0.0
        count = 0

        for text_seq, image_3sc, label, biip_seq, cap_seq in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCH}"):
            text_seq = text_seq.to(device)
            image_3sc = image_3sc.to(device)
            label = label.to(device)
            biip_seq = biip_seq.to(device)
            cap_seq = cap_seq.to(device)

            output = model(text_seq, image_3sc, biip_seq, cap_seq)
            loss = loss_func(output, label)

            optim_task_detection.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optim_task_detection.step()

            pred = output.argmax(1)
            corrects += pred.eq(label).sum().item()
            loss_total += loss.item() * label.size(0)
            count += label.size(0)

        acc_train = corrects / count
        loss_train = loss_total / count


        print("开始测试...")
        acc_test, loss_test, p1, r1, f1_1, p0, r0, f1_0, test_probs, test_labels, _ = test(model, test_loader,
                                                                                           return_features=False)

        scheduler.step()
        current_lr = optim_task_detection.param_groups[0]["lr"]
        lr_list.append(current_lr)


        training_history['train_loss'].append(float(loss_train))
        training_history['test_loss'].append(float(loss_test))
        training_history['train_acc'].append(float(acc_train))
        training_history['test_acc'].append(float(acc_test))
        training_history['precision_rumor'].append(float(p1))
        training_history['recall_rumor'].append(float(r1))
        training_history['f1_rumor'].append(float(f1_1))
        training_history['precision_non_rumor'].append(float(p0))
        training_history['recall_non_rumor'].append(float(r0))
        training_history['f1_non_rumor'].append(float(f1_0))


        if acc_test > best_acc:
            best_acc = acc_test
            save_checkpoint(epoch, model, optim_task_detection, best_acc, 'best_model')
            print(f"新的最佳模型已保存! 准确率: {best_acc:.4f}")
            plot_roc_pr_curves(test_labels, test_probs, "best_model_roc_pr")

        if (epoch + 1) % 10 == 0:
            save_training_history(training_history, f'training_history_epoch_{epoch + 1}')
            plot_training_curves(training_history, lr_list, f'training_curves_epoch_{epoch + 1}')

        print('---  TASK Detection  ---')
        print(f"EPOCH = {epoch + 1}\n"
              f"acc_train = {acc_train:.3f}\nacc_test = {acc_test:.3f}\n"
              f"loss_train = {loss_train:.3f}\nloss_test = {loss_test:.3f}")
        print('Rumor:   P {:.3f} R {:.3f} F1 {:.3f}'.format(p1, r1, f1_1))
        print('NonRumor: P {:.3f} R {:.3f} F1 {:.3f}'.format(p0, r0, f1_0))


    plot_roc_pr_curves(test_labels, test_probs, "final_epoch_roc_pr")

    best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{TIMESTAMP}.pth")
    if os.path.exists(best_model_path):
        print(f"\n[Info] 正在加载最佳模型权重以绘制 T-SNE: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[Info] 权重加载成功！开始提取特征...")
    else:
        print("\n[Warning] 未找到最佳模型文件，将使用最后一次训练的模型绘制 T-SNE")

    _, _, _, _, _, _, _, _, _, tsne_labels, tsne_feats = test(model, test_loader, return_features=True)
    if tsne_feats is not None:
        plot_tsne(tsne_feats, tsne_labels, "best_model_tsne")

    save_training_history(training_history, 'final_training_history')
    plot_training_curves(training_history, lr_list, 'final_training_curves')
    print("训练完成! 所有结果已保存。")

if __name__ == "__main__":
    train()