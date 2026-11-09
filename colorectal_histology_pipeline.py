# Colorectal Histology Classification with InceptionV3 + Explainable AI
# Recommended environment: Google Colab with a GPU runtime.
#
# Dataset: kagglehub.dataset_download("kmader/colorectal-histology-mnist")
# This script:
#   1. Downloads the dataset using KaggleHub.
#   2. Finds the ImageFolder-style class directory automatically.
#   3. Creates stratified train/validation/test splits (70/15/15).
#   4. Uses separate training and evaluation transforms.
#   5. Trains InceptionV3 in two stages (frozen head, then fine-tuning).
#   6. Evaluates using accuracy, precision, recall, F1, specificity, ROC-AUC.
#   7. Saves reports, predictions, plots, and the best checkpoint.
#   8. Generates Grad-CAM and Integrated Gradients explanations.
#   9. Optionally runs LIME (can take additional time).
#
# Important:
# - The Kather tile dataset does not provide patient/slide grouping metadata
#   in the usual ImageFolder layout. A stratified tile-level split is therefore
#   used here; it is NOT equivalent to an independent patient-level evaluation.
# - SHAP is not included in this stable baseline because the earlier notebook's
#   SHAP wrapper disabled gradients with torch.no_grad(), making gradient-based
#   SHAP invalid. Integrated Gradients is used as a reliable gradient-based
#   explanation instead.
# - This is a research/educational tissue-tile classifier, not a clinical tool.

# ========================= 1. INSTALL PACKAGES =========================
# Run this script in Google Colab. If dependencies are already installed,
# pip will only install/update what is needed.
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "kagglehub", "captum", "lime", "scikit-learn",
    "seaborn", "opencv-python-headless", "pandas",
    "matplotlib", "pillow"
])

# ========================= 2. IMPORTS AND CONFIGURATION =========================
import os
import random
import json
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- USER SETTINGS -------------------------
DATASET_HANDLE = "kmader/colorectal-histology-mnist"

SEED = 42
INPUT_SIZE = 299
BATCH_SIZE = 32
NUM_WORKERS = 2

# Training schedule. Adjust if you have limited GPU time.
STAGE1_EPOCHS = 6
STAGE2_EPOCHS = 12
PATIENCE = 5

# Learning rates
HEAD_LR = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4

# Explainability settings
RUN_GRADCAM = True
RUN_INTEGRATED_GRADIENTS = True
RUN_LIME = True          # Set False to skip the slower LIME step.
LIME_NUM_SAMPLES = 300   # Increase to 800-1500 for more stable LIME results.

# Output directory in Colab. Download it from the Files panel when finished.
OUTPUT_DIR = Path("/content/colorectal_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ImageNet normalization expected by pretrained InceptionV3 weights.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ========================= 3. REPRODUCIBILITY =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode can reduce performance, but helps reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print("=" * 75)
print("COLORECTAL HISTOLOGY CLASSIFICATION")
print("=" * 75)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not detected. In Colab, select Runtime > Change runtime type > T4 GPU.")

# ========================= 4. DOWNLOAD DATASET WITH KAGGLEHUB =========================
import kagglehub

print("\nDownloading dataset using KaggleHub...")
dataset_download_path = kagglehub.dataset_download(DATASET_HANDLE)
print("Path to dataset files:", dataset_download_path)

# ImageFolder expects:
# dataset_root/
#   class_1/
#       image1.tif
#   class_2/
#       image2.tif
#
# Search for the directory with the largest number of images directly
# inside its immediate class subdirectories.
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

def count_images_in_folder(folder):
    count = 0
    try:
        for item in folder.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                count += sum(
                    1 for f in item.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                )
    except PermissionError:
        return 0
    return count

def find_imagefolder_root(download_path):
    base = Path(download_path)
    candidates = []

    for current, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        current_path = Path(current)
        child_dirs = [
            current_path / d for d in dirs
            if not d.startswith(".")
        ]

        class_dirs = []
        total = 0
        for child in child_dirs:
            n = sum(
                1 for f in child.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if n > 0:
                class_dirs.append(child)
                total += n

        if len(class_dirs) >= 2 and total > 0:
            candidates.append((total, current_path, class_dirs))

    if not candidates:
        raise FileNotFoundError(
            "Could not find a folder containing class subfolders with images. "
            "Inspect the KaggleHub download directory printed above and update "
            "DATASET_ROOT manually if the dataset layout is different."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_total, best_root, best_classes = candidates[0]

    print("\nDetected ImageFolder root:", best_root)
    print("Detected image count:", best_total)
    print("Detected class folders:")
    for folder in sorted(best_classes):
        n = sum(
            1 for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"  {folder.name}: {n}")

    return str(best_root)

DATASET_ROOT = find_imagefolder_root(dataset_download_path)

# ========================= 5. TRANSFORMS AND DATA SPLITS =========================
train_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 90)),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

eval_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Separate dataset objects are important: train uses random augmentation,
# validation/test use deterministic transforms.
train_dataset_full = datasets.ImageFolder(DATASET_ROOT, transform=train_transforms)
eval_dataset_full = datasets.ImageFolder(DATASET_ROOT, transform=eval_transforms)

classes = train_dataset_full.classes
num_classes = len(classes)
targets = np.array(train_dataset_full.targets)
indices = np.arange(len(train_dataset_full))

if num_classes < 2:
    raise ValueError(f"Expected multiple tissue classes, found: {classes}")

print("\nClasses:", classes)
print("Total images:", len(train_dataset_full))

# 70% train, 15% validation, 15% test with stratification.
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=targets
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=targets[temp_idx]
)

train_dataset = Subset(train_dataset_full, train_idx.tolist())
val_dataset = Subset(eval_dataset_full, val_idx.tolist())
test_dataset = Subset(eval_dataset_full, test_idx.tolist())

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0)
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0)
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0)
)

def print_split_distribution(name, split_indices):
    split_labels = targets[split_indices]
    counts = np.bincount(split_labels, minlength=num_classes)
    print(f"\n{name} distribution:")
    for class_name, count in zip(classes, counts):
        print(f"  {class_name}: {count}")

print_split_distribution("Training", train_idx)
print_split_distribution("Validation", val_idx)
print_split_distribution("Testing", test_idx)

# Save split membership so the experiment can be reproduced.
split_df = pd.DataFrame({
    "index": np.concatenate([train_idx, val_idx, test_idx]),
    "split": (
        ["train"] * len(train_idx) +
        ["validation"] * len(val_idx) +
        ["test"] * len(test_idx)
    ),
    "label": [classes[targets[i]] for i in np.concatenate([train_idx, val_idx, test_idx])],
    "path": [train_dataset_full.samples[i][0] for i in np.concatenate([train_idx, val_idx, test_idx])]
})
split_df.to_csv(OUTPUT_DIR / "dataset_splits.csv", index=False)

# ========================= 6. MODEL =========================
def create_inception_model(number_of_classes):
    # Modern torchvision weights API.
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, aux_logits=True)

    # Replace the main and auxiliary classifiers.
    model.fc = nn.Linear(model.fc.in_features, number_of_classes)
    if model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(
            model.AuxLogits.fc.in_features, number_of_classes
        )
    return model

model = create_inception_model(num_classes).to(DEVICE)

def freeze_all_layers(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if model.AuxLogits is not None:
        for parameter in model.AuxLogits.fc.parameters():
            parameter.requires_grad = True

def unfreeze_late_inception_layers(model):
    # Fine-tune later Inception blocks plus both classification heads.
    for module_name in ["Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c"]:
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if model.AuxLogits is not None:
        for parameter in model.AuxLogits.fc.parameters():
            parameter.requires_grad = True

# InceptionV3 has auxiliary logits during training. In eval mode it returns
# a single logits tensor.
def get_main_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    return output

criterion = nn.CrossEntropyLoss()

# ========================= 7. TRAINING AND VALIDATION =========================
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

def run_one_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    # During Stage 1, keep BatchNorm running statistics fixed in frozen
    # layers. This avoids silently changing the pretrained backbone statistics.
    if training:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                trainable_bn_params = any(
                    parameter.requires_grad
                    for parameter in module.parameters(recurse=False)
                )
                if not trainable_bn_params:
                    module.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                outputs = model(images)

                if training and hasattr(outputs, "logits"):
                    main_logits = outputs.logits
                    aux_logits = outputs.aux_logits
                    loss = criterion(main_logits, labels)
                    if aux_logits is not None:
                        loss = loss + 0.4 * criterion(aux_logits, labels)
                else:
                    main_logits = get_main_logits(outputs)
                    loss = criterion(main_logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = main_logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_count += batch_size

    return total_loss / total_count, 100.0 * total_correct / total_count

GLOBAL_BEST_VAL_LOSS = float("inf")

def train_stage(stage_name, epochs, optimizer, scheduler, best_checkpoint_path):
    global GLOBAL_BEST_VAL_LOSS

    # Local best is used for early stopping/restoring this stage's best weights.
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    print(f"\n{'=' * 70}\n{stage_name}\n{'=' * 70}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, optimizer=None)

        scheduler.step(val_loss)

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.2f}% | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Preserve the best checkpoint across BOTH stages, rather than
        # allowing Stage 2 to overwrite a better Stage 1 model.
        if val_loss < GLOBAL_BEST_VAL_LOSS:
            GLOBAL_BEST_VAL_LOSS = val_loss
            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "classes": classes,
                "class_to_idx": train_dataset_full.class_to_idx,
                "input_size": INPUT_SIZE,
                "mean": MEAN,
                "std": STD,
                "architecture": "InceptionV3",
                "stage": stage_name,
                "best_val_loss": GLOBAL_BEST_VAL_LOSS
            }, best_checkpoint_path)
            print("  Saved globally best validation checkpoint.")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation loss did not improve.")
            break

    model.load_state_dict(best_state)
    return pd.DataFrame(history)

# Stage 1: train only classifier heads.
freeze_all_layers(model)
stage1_params = [p for p in model.parameters() if p.requires_grad]
optimizer_stage1 = torch.optim.AdamW(
    stage1_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY
)
scheduler_stage1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_stage1, mode="min", factor=0.3, patience=2
)

checkpoint_path = OUTPUT_DIR / "best_inceptionv3_checkpoint.pth"

history1 = train_stage(
    "Stage 1 - Frozen backbone",
    STAGE1_EPOCHS,
    optimizer_stage1,
    scheduler_stage1,
    checkpoint_path
)

# Stage 2: fine-tune later convolutional blocks with a lower learning rate.
unfreeze_late_inception_layers(model)

backbone_params = []
head_params = []
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        if name.startswith("fc.") or name.startswith("AuxLogits.fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

optimizer_stage2 = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR * 0.5}
    ],
    weight_decay=WEIGHT_DECAY
)
scheduler_stage2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_stage2, mode="min", factor=0.3, patience=2
)

history2 = train_stage(
    "Stage 2 - Fine-tuning later Inception blocks",
    STAGE2_EPOCHS,
    optimizer_stage2,
    scheduler_stage2,
    checkpoint_path
)

history_df = pd.concat([history1, history2], ignore_index=True)
history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

# Load the best checkpoint across both stages.
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ========================= 8. TRAINING CURVES =========================
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["train_accuracy"], label="Train accuracy")
    axes[1].plot(history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=250, bbox_inches="tight")
    plt.show()

plot_training_history(history_df)

# ========================= 9. TEST SET EVALUATION =========================
def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred, y_prob, paths = [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                logits = get_main_logits(model(images))
                probabilities = torch.softmax(logits, dim=1)

            predictions = logits.argmax(dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)

y_true, y_pred, y_prob = evaluate_model(model, test_loader)

overall_accuracy = accuracy_score(y_true, y_pred)
macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
balanced_acc = balanced_accuracy_score(y_true, y_pred)

try:
    macro_auc = roc_auc_score(
        y_true, y_prob, multi_class="ovr", average="macro"
    )
except ValueError:
    macro_auc = float("nan")

metrics = {
    "Accuracy": overall_accuracy,
    "Macro Precision": macro_precision,
    "Macro Recall": macro_recall,
    "Macro F1": macro_f1,
    "Weighted F1": weighted_f1,
    "Balanced Accuracy": balanced_acc,
    "Macro ROC-AUC (OvR)": macro_auc
}

print("\n" + "=" * 70)
print("FINAL TEST METRICS")
print("=" * 70)
for name, value in metrics.items():
    print(f"{name:25s}: {value:.4f}")

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

report_dict = classification_report(
    y_true, y_pred, target_names=classes,
    output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

print("\nClassification report:")
print(classification_report(
    y_true, y_pred, target_names=classes, digits=4, zero_division=0
))

# Per-class specificity = TN / (TN + FP), calculated one-vs-rest.
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
specificities = []
for class_idx in range(num_classes):
    tp = cm[class_idx, class_idx]
    fn = cm[class_idx, :].sum() - tp
    fp = cm[:, class_idx].sum() - tp
    tn = cm.sum() - (tp + fn + fp)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    specificities.append(specificity)

per_class_df = pd.DataFrame({
    "class": classes,
    "recall_sensitivity": [
        recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        for i in range(num_classes)
    ],
    "specificity": specificities
})
per_class_df.to_csv(OUTPUT_DIR / "per_class_sensitivity_specificity.csv", index=False)

# Save predictions and class probabilities.
predictions_df = pd.DataFrame({
    "true_index": y_true,
    "predicted_index": y_pred,
    "true_class": [classes[i] for i in y_true],
    "predicted_class": [classes[i] for i in y_pred],
    "correct": (y_true == y_pred)
})
for class_idx, class_name in enumerate(classes):
    predictions_df[f"prob_{class_name}"] = y_prob[:, class_idx]
predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# ========================= 10. CONFUSION MATRIX =========================
plt.figure(figsize=(11, 9))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=classes, yticklabels=classes
)
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=250, bbox_inches="tight")
plt.show()

row_sums = cm.sum(axis=1, keepdims=True)
normalized_cm = np.divide(
    cm.astype(float), row_sums,
    out=np.zeros_like(cm, dtype=float), where=row_sums != 0
)
plt.figure(figsize=(11, 9))
sns.heatmap(
    normalized_cm, annot=True, fmt=".2f", cmap="Blues",
    vmin=0, vmax=1, xticklabels=classes, yticklabels=classes
)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "normalized_confusion_matrix.png", dpi=250, bbox_inches="tight")
plt.show()

# ========================= 11. MULTICLASS ROC CURVES =========================
y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(classes):
    if len(np.unique(y_true_bin[:, i])) < 2:
        continue
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    class_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC={class_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves")
plt.legend(fontsize=8, loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=250, bbox_inches="tight")
plt.show()

# ========================= 12. EXPLAINABILITY HELPERS =========================
def denormalize_image(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(STD) + np.array(MEAN)
    return np.clip(image, 0, 1)

def get_test_examples(number=3):
    images, labels = [], []
    for batch_images, batch_labels in test_loader:
        for i in range(batch_images.size(0)):
            images.append(batch_images[i])
            labels.append(batch_labels[i].item())
            if len(images) >= number:
                return images, labels
    return images, labels

# ========================= 13. GRAD-CAM =========================
def run_gradcam(number_of_samples=3):
    print("\nGenerating Grad-CAM explanations...")
    model.eval()
    examples, labels = get_test_examples(number_of_samples)
    if not examples:
        print("No test examples available for Grad-CAM.")
        return

    target_layer = model.Mixed_7c
    gradcam = LayerGradCam(model, target_layer)

    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (image_tensor, true_label) in enumerate(zip(examples, labels)):
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            predicted_class = model(input_tensor).argmax(dim=1).item()

        attribution = gradcam.attribute(input_tensor, target=predicted_class)
        attribution = LayerAttribution.interpolate(
            attribution, (INPUT_SIZE, INPUT_SIZE),
            interpolate_mode="bilinear"
        )
        cam = attribution.squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        original = denormalize_image(image_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.55 * original + 0.45 * heatmap, 0, 1)

        axes[idx, 0].imshow(original)
        axes[idx, 0].set_title(f"Original\nTrue: {classes[true_label]}")
        axes[idx, 1].imshow(cam, cmap="jet")
        axes[idx, 1].set_title(f"Grad-CAM\nPred: {classes[predicted_class]}")
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay")

        for ax in axes[idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradcam_explanations.png", dpi=250, bbox_inches="tight")
    plt.show()

# ========================= 14. INTEGRATED GRADIENTS =========================
def run_integrated_gradients(number_of_samples=2):
    print("\nGenerating Integrated Gradients explanations...")
    model.eval()
    ig = IntegratedGradients(model)
    examples, labels = get_test_examples(number_of_samples)

    if not examples:
        print("No test examples available for Integrated Gradients.")
        return

    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (image_tensor, true_label) in enumerate(zip(examples, labels)):
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            predicted_class = model(input_tensor).argmax(dim=1).item()

        baseline = torch.zeros_like(input_tensor)
        attribution = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=predicted_class,
            n_steps=32
        )

        attr = attribution.squeeze(0).detach().cpu().numpy()
        attr_map = np.abs(attr).sum(axis=0)
        if attr_map.max() > attr_map.min():
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
        else:
            attr_map = np.zeros_like(attr_map)

        original = denormalize_image(image_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.55 * original + 0.45 * heatmap, 0, 1)

        axes[idx, 0].imshow(original)
        axes[idx, 0].set_title(f"Original\nTrue: {classes[true_label]}")
        axes[idx, 1].imshow(attr_map, cmap="hot")
        axes[idx, 1].set_title(f"Integrated Gradients\nPred: {classes[predicted_class]}")
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Overlay")

        for ax in axes[idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "integrated_gradients_explanations.png",
        dpi=250, bbox_inches="tight"
    )
    plt.show()

# ========================= 15. LIME (OPTIONAL, SLOWER) =========================
def run_lime(number_of_samples=1):
    print("\nGenerating LIME explanations...")
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    model.eval()
    examples, labels = get_test_examples(number_of_samples)
    if not examples:
        print("No test examples available for LIME.")
        return

    def predict_for_lime(images_np):
        # LIME supplies RGB images in [0, 1]. Convert to normalized model tensors.
        batch = []
        for image in images_np:
            pil_image = transforms.ToPILImage()(np.uint8(np.clip(image, 0, 1) * 255))
            tensor = eval_transforms(pil_image)
            batch.append(tensor)
        batch_tensor = torch.stack(batch).to(DEVICE)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                logits = get_main_logits(model(batch_tensor))
                probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    fig, axes = plt.subplots(len(examples), 2, figsize=(9, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (image_tensor, true_label) in enumerate(zip(examples, labels)):
        original = denormalize_image(image_tensor)
        explanation = explainer.explain_instance(
            original,
            predict_for_lime,
            top_labels=1,
            hide_color=0,
            num_samples=LIME_NUM_SAMPLES
        )
        predicted_class = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=False,
            num_features=8,
            hide_rest=False
        )

        axes[idx, 0].imshow(original)
        axes[idx, 0].set_title(f"Original\nTrue: {classes[true_label]}")
        axes[idx, 1].imshow(mark_boundaries(temp, mask))
        axes[idx, 1].set_title(f"LIME\nPred: {classes[predicted_class]}")
        for ax in axes[idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lime_explanations.png", dpi=250, bbox_inches="tight")
    plt.show()

# ========================= 16. RUN EXPLAINABILITY =========================
if RUN_GRADCAM:
    run_gradcam(number_of_samples=3)

if RUN_INTEGRATED_GRADIENTS:
    run_integrated_gradients(number_of_samples=2)

if RUN_LIME:
    try:
        run_lime(number_of_samples=1)
    except Exception as exc:
        print("LIME could not complete:", repr(exc))
        print("The trained model and other evaluation outputs have still been saved.")

# ========================= 17. SAVE FINAL MODEL AND SUMMARY =========================
final_model_path = OUTPUT_DIR / "colorectal_inceptionv3_final.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "input_size": INPUT_SIZE,
    "mean": MEAN,
    "std": STD,
    "architecture": "InceptionV3"
}, final_model_path)

summary = {
    "dataset": DATASET_HANDLE,
    "dataset_root": DATASET_ROOT,
    "classes": classes,
    "num_images": len(train_dataset_full),
    "train_images": len(train_idx),
    "validation_images": len(val_idx),
    "test_images": len(test_idx),
    "device": str(DEVICE),
    "metrics": metrics,
    "best_checkpoint": str(checkpoint_path),
    "final_model": str(final_model_path)
}
with open(OUTPUT_DIR / "experiment_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 75)
print("PIPELINE COMPLETE")
print("=" * 75)
print("Test accuracy:", f"{overall_accuracy * 100:.2f}%")
print("Macro F1:", f"{macro_f1:.4f}")
print("Output directory:", OUTPUT_DIR)
print("\nFiles generated:")
for file_path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file_path.name)

print(
    "\nReminder: report the split method and dataset limitations. "
    "Do not describe tile-level test performance as patient-level clinical validation."
)
