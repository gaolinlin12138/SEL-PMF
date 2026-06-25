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
from tqdm import tqdm
from weibo_dataset import FeatureDatasetV2
from weibo_model import TriBranchMSCAN
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

DEVICE = "cuda:0"
NUM_WORKER = 0
BATCH_SIZE = 32
LR = 5e-4
L2 = 1e-5
NUM_EPOCH = 100
WARMUP_EPOCHS = 5
MAX_GRAD_NORM = 1.0


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


SAVE_DIR = f"WEIBO/training_results_{TIMESTAMP}"
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
        'timestamp': TIMESTAMP
    }
    with open(os.path.join(RESULT_SAVE_DIR, f'training_config_{TIMESTAMP}.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def plot_training_curves(history, lr_list, filename):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(history['f1_rumor'], label='Rumor F1')
    plt.plot(history['f1_non_rumor'], label='Non-rumor F1')
    plt.title('F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(history['precision_rumor'], label='Rumor Precision')
    plt.plot(history['recall_rumor'], label='Rumor Recall')
    plt.title('Precision and Recall for Rumor')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(history['precision_non_rumor'], label='Non-rumor Precision')
    plt.plot(history['recall_non_rumor'], label='Non-rumor Recall')
    plt.title('Precision and Recall for Non-rumor')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(lr_list, label="Learning Rate", color="orange")
    plt.title("Learning Rate Schedule (Warmup + Cosine)")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    filename_with_timestamp = f"{filename.split('.')[0]}_{TIMESTAMP}.png"
    plt.savefig(os.path.join(RESULT_SAVE_DIR, filename_with_timestamp), dpi=300, bbox_inches='tight')
    plt.close()

def get_scheduler(optimizer, num_epochs, warmup_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    device = torch.device(DEVICE)
    loss_func = torch.nn.CrossEntropyLoss()

    count = 0
    loss_total = 0.0
    y_all = []
    yhat_all = []

    for text_seq, image_3sc, label, biip_seq, cap_seq in tqdm(test_loader, desc="Testing"):
        text_seq = text_seq.to(device)      # [B,150,1536]
        image_3sc = image_3sc.to(device)    # [B,3,1536]
        label = label.to(device)            # [B]
        biip_seq = biip_seq.to(device)      # [B,150,1536]
        cap_seq = cap_seq.to(device)        # [B,150,1536]

        output = model(text_seq, image_3sc, biip_seq, cap_seq)
        loss = loss_func(output, label)
        pred = output.argmax(1)

        loss_total += loss.item() * label.size(0)
        count += label.size(0)
        yhat_all.append(pred.detach().cpu().numpy())
        y_all.append(label.detach().cpu().numpy())

    loss_test = loss_total / count
    yhat_all = np.concatenate(yhat_all, 0)
    y_all = np.concatenate(y_all, 0)

    acc = accuracy_score(y_all, yhat_all)
    f1_1 = f1_score(y_all, yhat_all, pos_label=1)
    r1 = recall_score(y_all, yhat_all, pos_label=1)
    p1 = precision_score(y_all, yhat_all, pos_label=1)

    f1_0 = f1_score(y_all, yhat_all, pos_label=0)
    r0 = recall_score(y_all, yhat_all, pos_label=0)
    p0 = precision_score(y_all, yhat_all, pos_label=0)
    return acc, loss_test, p1, r1, f1_1, p0, r0, f1_0

def train():
    print(f"本次训练时间戳: {TIMESTAMP}")
    print(f"结果保存路径: {SAVE_DIR}")
    save_config()

    device = torch.device(DEVICE)
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH

    dataset_dir = 'Qwen_weibo_dataset_all'
    dataset_prefix = 'weibo_'


    train_text   = f"{dataset_dir}/{dataset_prefix}train_text_embed.npy"
    train_image  = f"{dataset_dir}/{dataset_prefix}train_image_embed.npy"
    train_label  = f"{dataset_dir}/{dataset_prefix}train_label.npy"
    train_biip   = f"{dataset_dir}/{dataset_prefix}train_text_BIs_IPs_embed.npy"
    train_cap    = f"{dataset_dir}/{dataset_prefix}train_image_caption_embed.npy"


    test_text    = f"{dataset_dir}/{dataset_prefix}test_text_embed.npy"
    test_image   = f"{dataset_dir}/{dataset_prefix}test_image_embed.npy"
    test_label   = f"{dataset_dir}/{dataset_prefix}test_label.npy"
    test_biip    = f"{dataset_dir}/{dataset_prefix}test_text_BIs_IPs_embed.npy"
    test_cap     = f"{dataset_dir}/{dataset_prefix}test_image_caption_embed.npy"

    # ---------------- DataLoader ----------------
    train_set = FeatureDatasetV2(train_text, train_image, train_label, train_biip, train_cap)
    test_set  = FeatureDatasetV2(test_text,  test_image,  test_label,  test_biip,  test_cap)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=NUM_WORKER,
                              shuffle=True, drop_last=True, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, num_workers=NUM_WORKER,
                              shuffle=False, drop_last=False, pin_memory=True)


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

    # ==================== Epoch Loop ====================
    for epoch in range(num_epoch):
        model.train()
        corrects = 0
        loss_total = 0.0
        count = 0

        for text_seq, image_3sc, label, biip_seq, cap_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCH}"):
            text_seq = text_seq.to(device)      # [B,150,1536]
            image_3sc = image_3sc.to(device)    # [B,3,1536]
            label = label.to(device)            # [B]
            biip_seq = biip_seq.to(device)      # [B,150,1536]
            cap_seq = cap_seq.to(device)        # [B,150,1536]
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
        acc_test, loss_test, p1, r1, f1_1, p0, r0, f1_0 = test(model, test_loader)


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


        if (epoch + 1) % 10 == 0:
            save_training_history(training_history, f'training_history_epoch_{epoch + 1}')
            plot_training_curves(training_history, lr_list, f'training_curves_epoch_{epoch + 1}')

        print('---  TASK Detection  ---')
        print(f"EPOCH = {epoch + 1}\n"
              f"acc_train = {acc_train:.3f}\nacc_test = {acc_test:.3f}\n"
              f"loss_train = {loss_train:.3f}\nloss_test = {loss_test:.3f}")
        print('Rumor:   P {:.3f} R {:.3f} F1 {:.3f}'.format(p1, r1, f1_1))
        print('NonRumor: P {:.3f} R {:.3f} F1 {:.3f}'.format(p0, r0, f1_0))

    save_training_history(training_history, 'final_training_history')
    plot_training_curves(training_history, lr_list, 'final_training_curves')
    print("训练完成! 所有结果已保存。")

if __name__ == "__main__":
    train()
