import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import S1GFloodDataset
from model import SiameseUNet


DATASET_PATH = r"D:\Multimodal_Project\dataset\S1GFloods"


def read_split(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"{filename} is missing. Run prepare_data.py first."
        )

    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    print("=" * 60)
    print("MULTIMODAL FLOOD MODEL PIPELINE TEST")
    print("=" * 60)

    # 1. Verify dataset directories
    for folder in ["A", "B", "Label"]:
        path = os.path.join(DATASET_PATH, folder)
        assert os.path.isdir(path), f"Missing directory: {path}"

    print("[PASS] Dataset directories found")

    # 2. Verify data splits
    train_files = read_split("train.txt")
    val_files = read_split("val.txt")
    test_files = read_split("test.txt")

    print("Train:", len(train_files))
    print("Validation:", len(val_files))
    print("Test:", len(test_files))

    assert len(train_files) > 0
    assert len(val_files) > 0
    assert len(test_files) > 0

    assert not (set(train_files) & set(val_files))
    assert not (set(train_files) & set(test_files))
    assert not (set(val_files) & set(test_files))

    print("[PASS] Split files are nonempty and have no overlap")

    # 3. Load one training sample as a batch
    dataset = S1GFloodDataset(DATASET_PATH, train_files)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    before, after, mask = next(iter(loader))

    print("\nBefore:", tuple(before.shape))
    print("After: ", tuple(after.shape))
    print("Mask:  ", tuple(mask.shape))

    assert before.ndim == 4
    assert after.shape == before.shape
    assert mask.shape[0] == before.shape[0]
    assert mask.shape[1] == 1
    assert mask.shape[2:] == before.shape[2:]

    print("[PASS] DataLoader batch loaded correctly")

    # 4. Initialize model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    model = SiameseUNet().to(device)
    model.train()

    before = before.to(device)
    after = after.to(device)
    mask = mask.to(device)

    # 5. Forward pass
    model.zero_grad(set_to_none=True)

    prediction = model(before, after)

    print("Prediction:", tuple(prediction.shape))

    assert prediction.shape == mask.shape, (
        f"Output {prediction.shape} does not match mask {mask.shape}"
    )

    assert torch.isfinite(prediction).all(), (
        "Model output contains NaN or infinity"
    )

    print("[PASS] Model forward pass")

    # 6. Calculate loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(prediction, mask)

    print("BCE loss:", loss.item())

    assert torch.isfinite(loss), "Loss is not finite"

    # 7. Backpropagation
    loss.backward()

    has_gradient = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )

    assert has_gradient, "No valid model gradients found"

    print("[PASS] Loss calculation")
    print("[PASS] Backpropagation")

    print("\n" + "=" * 60)
    print("SUCCESS: DATA-TO-MODEL PIPELINE WORKS")
    print("=" * 60)


if __name__ == "__main__":
    main()