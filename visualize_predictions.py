import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from dataset import S1GFloodDataset
from model import SiameseUNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:\Multimodal_Project\dataset\S1GFloods"
MODEL_PATH = "best_s1gflood_model.pth"
OUTPUT_DIR = "predictions"

NUM_SAMPLES = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# READ TEST SPLIT
# ============================================================

with open("test.txt", "r") as f:
    test_files = [
        line.strip()
        for line in f
        if line.strip()
    ]


# ============================================================
# LOAD MODEL
# ============================================================

model = SiameseUNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")
print("Device:", device)


# ============================================================
# DATASET
# ============================================================

dataset = S1GFloodDataset(
    DATASET_PATH,
    test_files
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

with torch.no_grad():

    for i in range(min(NUM_SAMPLES, len(dataset))):

        before, after, target = dataset[i]

        # Add batch dimension
        before_input = before.unsqueeze(0).to(device)
        after_input = after.unsqueeze(0).to(device)

        # Prediction
        prediction = model(
            before_input,
            after_input
        )

        prediction = torch.sigmoid(
            prediction
        )

        prediction = (
            prediction > 0.5
        ).float()

        # Remove batch/channel dimensions
        before_img = before.squeeze().numpy()
        after_img = after.squeeze().numpy()
        target_img = target.squeeze().numpy()
        prediction_img = prediction.squeeze().cpu().numpy()

        # ====================================================
        # SAVE INDIVIDUAL MASK
        # ====================================================

        mask_path = os.path.join(
            OUTPUT_DIR,
            f"prediction_{i+1}.png"
        )

        mask_uint8 = (
            prediction_img * 255
        ).astype(np.uint8)

        Image.fromarray(
            mask_uint8
        ).save(mask_path)

        # ====================================================
        # CREATE 4-PANEL VISUALIZATION
        # ====================================================

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        axes[0].imshow(
            before_img,
            cmap="gray"
        )
        axes[0].set_title("Before Flood")
        axes[0].axis("off")

        axes[1].imshow(
            after_img,
            cmap="gray"
        )
        axes[1].set_title("After Flood")
        axes[1].axis("off")

        axes[2].imshow(
            target_img,
            cmap="gray"
        )
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        axes[3].imshow(
            prediction_img,
            cmap="gray"
        )
        axes[3].set_title("Model Prediction")
        axes[3].axis("off")

        fig.suptitle(
            f"S1GFlood Prediction - Sample {i+1}",
            fontsize=14
        )

        plt.tight_layout()

        figure_path = os.path.join(
            OUTPUT_DIR,
            f"comparison_{i+1}.png"
        )

        plt.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"[{i+1}/{min(NUM_SAMPLES, len(dataset))}] "
            f"{test_files[i]}"
        )


print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print("Saved to:", os.path.abspath(OUTPUT_DIR))