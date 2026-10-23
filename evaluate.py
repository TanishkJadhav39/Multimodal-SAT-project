import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import S1GFloodDataset
from model import SiameseUNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:\Multimodal_Project\dataset\S1GFloods"
MODEL_PATH = "best_s1gflood_model.pth"

BATCH_SIZE = 4
NUM_WORKERS = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# READ TEST SPLIT
# ============================================================

def read_split(filename):

    with open(filename, "r") as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]


test_files = read_split("test.txt")

print("=" * 60)
print("S1GFLOOD TEST EVALUATION")
print("=" * 60)

print("Device:", device)
print("Test samples:", len(test_files))


# ============================================================
# DATASET
# ============================================================

test_dataset = S1GFloodDataset(
    DATASET_PATH,
    test_files
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

model = SiameseUNet().to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

print("Checkpoint loaded.")
print("Saved epoch:", checkpoint["epoch"])
print("Saved validation loss:", checkpoint["val_loss"])
print("Saved validation IoU:", checkpoint["val_iou"])
print("Saved validation Dice:", checkpoint["val_dice"])

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model weights loaded successfully.")


# ============================================================
# METRICS
# ============================================================

total_intersection = 0.0
total_union = 0.0

total_tp = 0.0
total_fp = 0.0
total_fn = 0.0

total_dice = 0.0
num_batches = 0


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for before, after, target in tqdm(
        test_loader,
        desc="Testing"
    ):

        before = before.to(device)
        after = after.to(device)
        target = target.to(device)

        prediction = model(
            before,
            after
        )

        prediction = torch.sigmoid(
            prediction
        )

        prediction = (
            prediction > 0.5
        ).float()

        # --------------------------------------------
        # IoU
        # --------------------------------------------

        intersection = (
            prediction * target
        ).sum()

        union = (
            prediction
            +
            target
            -
            prediction * target
        ).sum()

        total_intersection += intersection.item()
        total_union += union.item()

        # --------------------------------------------
        # TP / FP / FN
        # --------------------------------------------

        tp = (
            prediction * target
        ).sum()

        fp = (
            prediction * (1 - target)
        ).sum()

        fn = (
            (1 - prediction) * target
        ).sum()

        total_tp += tp.item()
        total_fp += fp.item()
        total_fn += fn.item()

        # --------------------------------------------
        # Dice
        # --------------------------------------------

        dice = (
            (2 * intersection + 1e-6)
            /
            (prediction.sum() + target.sum() + 1e-6)
        )

        total_dice += dice.item()
        num_batches += 1


# ============================================================
# FINAL METRICS
# ============================================================

iou = (
    total_intersection
    /
    (total_union + 1e-6)
)

dice = (
    total_dice
    /
    num_batches
)

precision = (
    total_tp
    /
    (total_tp + total_fp + 1e-6)
)

recall = (
    total_tp
    /
    (total_tp + total_fn + 1e-6)
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test IoU       : {iou:.4f}")
print(f"Test Dice      : {dice:.4f}")
print(f"Test Precision : {precision:.4f}")
print(f"Test Recall    : {recall:.4f}")

print("=" * 60)
print("TESTING COMPLETE")
print("=" * 60)