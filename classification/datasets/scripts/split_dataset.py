"""
Dataset Splitter — splits fragments into train/val/test sets (70/15/15).

Works with the subfolder structure:
  datasets/fragments/<type>Fragments/
    ├── *.bin files
    └── labels_<type>.csv

Usage:
  python datasets/scripts/split_dataset.py
  python datasets/scripts/split_dataset.py --fragments datasets/fragments --output datasets
  python datasets/scripts/split_dataset.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import os
import sys
import csv
import shutil
import argparse
import random
from collections import defaultdict, Counter


def load_all_fragments(fragments_dir):
    """
    Load fragment entries from all subfolder CSVs.
    
    Returns:
        list of (fragment_name, label, subfolder_name) tuples
    """
    rows = []
    for folder_name in sorted(os.listdir(fragments_dir)):
        folder_path = os.path.join(fragments_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Find CSV file in this folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            # No CSV — infer label from folder name (e.g., "pdfFragments" → "pdf")
            label = folder_name.replace('Fragments', '')
            bin_files = [f for f in os.listdir(folder_path)
                         if not f.startswith('.') and os.path.isfile(os.path.join(folder_path, f))]
            for fname in bin_files:
                rows.append((fname, label, folder_name))
            continue

        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get('fragment_name', '')
                    label = row.get('label', '')
                    if fname and label:
                        rows.append((fname, label, folder_name))

    return rows


def stratified_split(rows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split rows into train/val/test sets, stratified by label."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    random.seed(seed)

    by_label = defaultdict(list)
    for entry in rows:
        by_label[entry[1]].append(entry)

    train, val, test = [], [], []

    for label, items in sorted(by_label.items()):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def save_split(entries, fragments_dir, output_dir, split_name):
    """
    Copy fragment .bin files into the split directory and write a unified mapping CSV.
    Maintains subfolder structure inside each split.
    """
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    mapping_path = os.path.join(split_dir, "fragment_mapping.csv")
    copied = 0

    with open(mapping_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fragment_name', 'label', 'subfolder'])

        for fname, label, subfolder in entries:
            src = os.path.join(fragments_dir, subfolder, fname)
            # Keep flat inside split dir for easy loading
            dst = os.path.join(split_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                writer.writerow([fname, label, subfolder])
                copied += 1

    print(f"  {split_name}: {copied} fragments → {split_dir}")
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Split fragments into train/val/test sets (stratified by file type)"
    )
    parser.add_argument('--fragments', default='datasets/fragments',
                        help='Directory containing fragment subfolders (default: datasets/fragments)')
    parser.add_argument('--output', default='datasets',
                        help='Base output directory (train/, val/, test/ created inside)')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                        help='Training set ratio (default: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if not os.path.isdir(args.fragments):
        print(f"❌ Directory not found: {args.fragments}")
        sys.exit(1)

    print(f"🔧 Split Configuration:")
    print(f"   Fragments: {args.fragments}")
    print(f"   Output:    {args.output}")
    print(f"   Ratios:    train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print()

    rows = load_all_fragments(args.fragments)
    print(f"📂 Loaded {len(rows)} fragment entries")

    # Show class distribution
    dist = Counter(label for _, label, _ in rows)
    for label, count in sorted(dist.items()):
        print(f"   {label}: {count}")
    print()

    train, val, test = stratified_split(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"📊 Split results:")
    save_split(train, args.fragments, args.output, "train")
    save_split(val, args.fragments, args.output, "val")
    save_split(test, args.fragments, args.output, "test")

    total = len(train) + len(val) + len(test)
    print(f"\n✅ Split complete!")
    print(f"   Train: {len(train)} ({len(train)/total*100:.1f}%)")
    print(f"   Val:   {len(val)} ({len(val)/total*100:.1f}%)")
    print(f"   Test:  {len(test)} ({len(test)/total*100:.1f}%)")


if __name__ == "__main__":
    main()
