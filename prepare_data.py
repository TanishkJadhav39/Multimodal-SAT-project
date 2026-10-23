import os
import random


DATASET_PATH = r"D:\Multimodal_Project\dataset\S1GFloods"

A_DIR = os.path.join(DATASET_PATH, "A")
B_DIR = os.path.join(DATASET_PATH, "B")
LABEL_DIR = os.path.join(DATASET_PATH, "Label")


print("Checking dataset...")


# Get PNG files
all_files = [
    f for f in os.listdir(A_DIR)
    if f.lower().endswith(".png")
]


# Numerical sorting
all_files.sort(
    key=lambda x: int(
        os.path.splitext(x)[0].split("_")[-1]
    )
)


print("Total A images:", len(all_files))


# Check matching files
valid_files = []

for filename in all_files:

    a_path = os.path.join(A_DIR, filename)
    b_path = os.path.join(B_DIR, filename)
    label_path = os.path.join(LABEL_DIR, filename)

    if (
        os.path.exists(a_path)
        and os.path.exists(b_path)
        and os.path.exists(label_path)
    ):
        valid_files.append(filename)


print("Matched pairs:", len(valid_files))


# Shuffle
random.seed(42)
random.shuffle(valid_files)


# Split
train_files = valid_files[:4300]
val_files = valid_files[4300:4830]
test_files = valid_files[4830:5360]


print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))


# Save
with open("train.txt", "w") as f:
    f.write("\n".join(train_files))

with open("val.txt", "w") as f:
    f.write("\n".join(val_files))

with open("test.txt", "w") as f:
    f.write("\n".join(test_files))


print("SUCCESS: Dataset split files created.")