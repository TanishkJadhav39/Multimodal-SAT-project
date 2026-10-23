import os
import torch
from pathlib import Path
from dataset import S1GFloodDataset


def find_dataset_root(project_dir):
    """Find the folder containing A, B, and Label directories."""
    candidates = []

    for current, dirs, _ in os.walk(project_dir):
        dirs[:] = [
            d for d in dirs
            if d not in {".venv", ".git", "__pycache__"}
        ]

        current_path = Path(current)

        if all((current_path / name).is_dir()
               for name in ["A", "B", "Label"]):

            a_files = {
                f.name for f in (current_path / "A").iterdir()
                if f.is_file()
            }
            b_files = {
                f.name for f in (current_path / "B").iterdir()
                if f.is_file()
            }
            label_files = {
                f.name for f in (current_path / "Label").iterdir()
                if f.is_file()
            }

            matched = a_files & b_files & label_files

            if matched:
                candidates.append((len(matched), current_path, matched))

    if not candidates:
        raise FileNotFoundError(
            "Could not find a dataset folder containing "
            "matching A, B, and Label directories."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, root, files = candidates[0]

    return root, sorted(files)


def main():
    project_dir = Path(__file__).resolve().parent

    print("=" * 50)
    print("S1GFloodDataset Test")
    print("=" * 50)

    # Automatically locate the dataset
    root_dir, files = find_dataset_root(project_dir)

    print(f"Dataset root: {root_dir}")
    print(f"Matched files: {len(files)}")

    # Create dataset
    dataset = S1GFloodDataset(
        root_dir=str(root_dir),
        files=files
    )

    print(f"Dataset length: {len(dataset)}")

    # Test several samples
    samples_to_test = min(5, len(dataset))

    for i in range(samples_to_test):
        before, after, label = dataset[i]

        print(f"\nSample {i + 1}")
        print(f"Before shape: {before.shape}")
        print(f"After shape:  {after.shape}")
        print(f"Label shape:  {label.shape}")

        assert before.ndim == 3, "Before image must be 3D"
        assert after.ndim == 3, "After image must be 3D"
        assert label.ndim == 3, "Label mask must be 3D"

        assert before.shape[0] == 1
        assert after.shape[0] == 1
        assert label.shape[0] == 1

        assert before.shape[-2:] == after.shape[-2:]
        assert before.shape[-2:] == label.shape[-2:]

        assert torch.isfinite(before).all()
        assert torch.isfinite(after).all()
        assert torch.isfinite(label).all()

        assert before.min() >= 0 and before.max() <= 1
        assert after.min() >= 0 and after.max() <= 1

        unique_labels = torch.unique(label)
        assert all(
            value.item() in (0.0, 1.0)
            for value in unique_labels
        ), f"Unexpected mask values: {unique_labels}"

        print(f"Before range: {before.min():.3f} to {before.max():.3f}")
        print(f"After range:  {after.min():.3f} to {after.max():.3f}")
        print(f"Label values: {unique_labels.tolist()}")

    print("\nSUCCESS: Dataset loading and basic checks passed.")


if __name__ == "__main__":
    main()