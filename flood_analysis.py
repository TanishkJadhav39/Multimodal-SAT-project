import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from dataset import S1GFloodDataset
from model import SiameseUNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:\Multimodal_Project\dataset\S1GFloods"
MODEL_PATH = "best_s1gflood_model.pth"

OUTPUT_CSV = "flood_analysis_results.csv"

BATCH_SIZE = 4
NUM_WORKERS = 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SEVERITY CLASSIFICATION
# ============================================================

def get_severity(percentage):

    if percentage < 5:
        return "LOW"

    elif percentage < 20:
        return "MODERATE"

    elif percentage < 40:
        return "HIGH"

    else:
        return "SEVERE"


# ============================================================
# READ TEST FILES
# ============================================================

def read_split(filename):

    with open(filename, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


test_files = read_split("test.txt")


# ============================================================
# START
# ============================================================

print("=" * 60)
print("S1GFLOOD FLOOD ANALYSIS")
print("=" * 60)

print("Device:", device)
print("Test samples:", len(test_files))


# ============================================================
# DATASET
# ============================================================

dataset = S1GFloodDataset(
    DATASET_PATH,
    test_files
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

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# ANALYSIS
# ============================================================

results = []

total_pixels = 0
total_flooded_pixels = 0

severity_counts = {
    "LOW": 0,
    "MODERATE": 0,
    "HIGH": 0,
    "SEVERE": 0
}


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    for index in tqdm(
        range(len(dataset)),
        desc="Analysing test images"
    ):

        before, after, target = dataset[index]

        before = before.unsqueeze(0).to(device)
        after = after.unsqueeze(0).to(device)

        # --------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------

        prediction = model(
            before,
            after
        )

        probability = torch.sigmoid(
            prediction
        )

        mask = (
            probability > 0.5
        ).float()

        # --------------------------------------------
        # FLOOD PIXELS
        # --------------------------------------------

        flood_pixels = int(
            mask.sum().item()
        )

        pixels = int(
            mask.numel()
        )

        percentage = (
            flood_pixels / pixels
        ) * 100

        severity = get_severity(
            percentage
        )

        # --------------------------------------------
        # CONFIDENCE
        # --------------------------------------------

        confidence = float(
            probability.mean().item()
        )

        # --------------------------------------------
        # STORE
        # --------------------------------------------

        filename = test_files[index]

        results.append({
            "filename": filename,
            "flooded_pixels": flood_pixels,
            "total_pixels": pixels,
            "flood_percentage": percentage,
            "severity": severity,
            "average_prediction_probability": confidence
        })

        total_flooded_pixels += flood_pixels
        total_pixels += pixels

        severity_counts[severity] += 1


# ============================================================
# OVERALL STATISTICS
# ============================================================

overall_percentage = (
    total_flooded_pixels /
    total_pixels
) * 100

average_percentage = np.mean([
    r["flood_percentage"]
    for r in results
])

maximum_percentage = np.max([
    r["flood_percentage"]
    for r in results
])

minimum_percentage = np.min([
    r["flood_percentage"]
    for r in results
])


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "flooded_pixels",
            "total_pixels",
            "flood_percentage",
            "severity",
            "average_prediction_probability"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FLOOD ANALYSIS RESULTS")
print("=" * 60)

print(
    f"Total test images       : {len(results)}"
)

print(
    f"Average flooded area    : {average_percentage:.2f}%"
)

print(
    f"Minimum flooded area    : {minimum_percentage:.2f}%"
)

print(
    f"Maximum flooded area    : {maximum_percentage:.2f}%"
)

print(
    f"Overall flooded pixels  : {overall_percentage:.2f}%"
)

print("\nSeverity distribution:")

for severity, count in severity_counts.items():

    percentage = (
        count / len(results)
    ) * 100

    print(
        f"{severity:10s}: "
        f"{count:3d} images "
        f"({percentage:.2f}%)"
    )

print("\nResults saved to:")
print(
    os.path.abspath(OUTPUT_CSV)
)

print("=" * 60)
print("FLOOD ANALYSIS COMPLETE")
print("=" * 60)