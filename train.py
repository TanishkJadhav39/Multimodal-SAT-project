import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import S1GFloodDataset
from model import SiameseUNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:\Multimodal_Project\dataset\S1GFloods"

BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-4

NUM_WORKERS = 0

MODEL_PATH = "best_s1gflood_model.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("S1GFLOOD MULTIMODAL FLOOD SEGMENTATION")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# LOAD FILE SPLITS
# ============================================================

def read_split(filename):

    with open(filename, "r") as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]


train_files = read_split("train.txt")
val_files = read_split("val.txt")
test_files = read_split("test.txt")


print("Training samples:", len(train_files))
print("Validation samples:", len(val_files))
print("Test samples:", len(test_files))


# ============================================================
# DATASETS
# ============================================================

train_dataset = S1GFloodDataset(
    DATASET_PATH,
    train_files
)

val_dataset = S1GFloodDataset(
    DATASET_PATH,
    val_files
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = SiameseUNet().to(device)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(prediction, target):

    prediction = torch.sigmoid(prediction)

    smooth = 1.0

    intersection = (
        prediction * target
    ).sum(dim=(1, 2, 3))

    denominator = (
        prediction.sum(dim=(1, 2, 3))
        +
        target.sum(dim=(1, 2, 3))
    )

    dice = (
        (2.0 * intersection + smooth)
        /
        (denominator + smooth)
    )

    return 1.0 - dice.mean()


def combined_loss(prediction, target):

    bce = bce_loss(
        prediction,
        target
    )

    dice = dice_loss(
        prediction,
        target
    )

    return bce + dice


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# METRICS
# ============================================================

def calculate_iou(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (
        prediction > 0.5
    ).float()

    intersection = (
        prediction * target
    ).sum(dim=(1, 2, 3))

    union = (
        prediction
        +
        target
        -
        prediction * target
    ).sum(dim=(1, 2, 3))

    iou = (
        (intersection + 1e-7)
        /
        (union + 1e-7)
    )

    return iou.mean().item()


def calculate_dice(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (
        prediction > 0.5
    ).float()

    intersection = (
        prediction * target
    ).sum(dim=(1, 2, 3))

    dice = (
        (2 * intersection + 1e-7)
        /
        (
            prediction.sum(dim=(1, 2, 3))
            +
            target.sum(dim=(1, 2, 3))
            +
            1e-7
        )
    )

    return dice.mean().item()


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [TRAIN]"
    )

    for before, after, target in train_bar:

        before = before.to(device)
        after = after.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        prediction = model(
            before,
            after
        )

        loss = combined_loss(
            prediction,
            target
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss /= len(train_loader)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0

    with torch.no_grad():

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [VAL]"
        )

        for before, after, target in val_bar:

            before = before.to(device)
            after = after.to(device)
            target = target.to(device)

            prediction = model(
                before,
                after
            )

            loss = combined_loss(
                prediction,
                target
            )

            val_loss += loss.item()

            val_iou += calculate_iou(
                prediction,
                target
            )

            val_dice += calculate_dice(
                prediction,
                target
            )

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val IoU    : {val_iou:.4f}"
    )

    print(
        f"Val Dice   : {val_dice:.4f}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_dice
            },
            MODEL_PATH
        )

        print(
            f"BEST MODEL SAVED → {MODEL_PATH}"
        )


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(
    f"Best validation loss: {best_val_loss:.4f}"
)
print(
    f"Model saved as: {MODEL_PATH}"
)