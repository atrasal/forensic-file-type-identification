"""
Fragment Clustering Script for Digital Forensics ML Project.

This script clusters fragments belonging to the same original file by
extracting byte frequency features and applying DBSCAN clustering
using Cosine Similarity.

Improvements over the original:
  - File-type-aware pre-grouping from fragment filenames
  - Richer feature set: byte freq + coarse bigram + transition matrix +
    entropy + ASCII ratio + distinct ratio + length
  - Adaptive DBSCAN eps per file-type group with silhouette-guided K-Means fallback
  - Ground-truth file-type label saved in output JSON
"""

import argparse
import os
import re
import json
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Support optional tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, desc=None: iterable


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def compute_entropy(byte_array):
    """Shannon entropy of a byte array (0–8 bits)."""
    counts = np.bincount(byte_array, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / byte_array.size
    return -np.sum(probs * np.log2(probs)) if probs.size > 0 else 0.0


def compute_entropy_variance(byte_array, block_size=256):
    """Variance of per-block entropy — measures entropy consistency."""
    n_blocks = max(1, byte_array.size // block_size)
    entropies = []
    for i in range(n_blocks):
        block = byte_array[i * block_size : (i + 1) * block_size]
        if block.size > 0:
            entropies.append(compute_entropy(block))
    return np.var(entropies) if len(entropies) > 1 else 0.0


def extract_features(file_path):
    """
    Extracts a rich feature vector for a fragment file.

    Feature dimensions:
      - Byte frequency histogram        (256)
      - Coarse byte-pair transitions     (256)  — 16×16 buckets
      - Shannon entropy                  (  1)
      - Entropy variance across blocks   (  1)
      - Distinct byte ratio              (  1)
      - ASCII printable ratio            (  1)
      - Length feature (log-scaled)       (  1)
    Total: 517 dimensions
    """
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    if len(file_bytes) == 0:
        return np.zeros(517, dtype=np.float64)

    byte_array = np.frombuffer(file_bytes, dtype=np.uint8)

    # 1. Byte frequency features (256 dimensions)
    byte_counts = np.zeros(256, dtype=np.float64)
    unique_bytes, counts = np.unique(byte_array, return_counts=True)
    byte_counts[unique_bytes] = counts

    # 2. Coarse byte-pair transition matrix (16×16 = 256 dimensions)
    #    Maps each byte to one of 16 buckets (upper nibble) then counts
    #    transitions between consecutive buckets. Much less lossy than the
    #    old XOR-shift approach.
    bigram_counts = np.zeros(256, dtype=np.float64)
    if byte_array.size > 1:
        hi_nibbles = byte_array >> 4                      # 0..15
        bucket_pairs = hi_nibbles[:-1].astype(np.uint16) * 16 + hi_nibbles[1:]
        unique_bigrams, bigram_freq = np.unique(bucket_pairs, return_counts=True)
        bigram_counts[unique_bigrams] = bigram_freq

    # 3. Statistical meta-features
    entropy = compute_entropy(byte_array)
    entropy_var = compute_entropy_variance(byte_array)
    distinct_ratio = np.count_nonzero(byte_counts) / 256.0
    ascii_ratio = np.sum((byte_array >= 0x20) & (byte_array <= 0x7E)) / byte_array.size
    length_feature = np.log1p(byte_array.size) / np.log1p(1024 * 1024)

    # Assemble and L2-normalize
    feature_vector = np.concatenate([
        byte_counts,
        bigram_counts,
        [entropy, entropy_var, distinct_ratio, ascii_ratio, length_feature],
    ])
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm

    return feature_vector


def prepare_feature_matrix(file_paths):
    """Constructs the feature matrix for the fragment files."""
    features = []
    for file_path in tqdm(file_paths, desc="Extracting features"):
        features.append(extract_features(file_path))
    return np.vstack(features)


# ---------------------------------------------------------------------------
# File-type inference from filename
# ---------------------------------------------------------------------------

# Ordered by specificity — longer / more specific patterns first to avoid
# e.g. "js" matching before "json".
_TYPE_PATTERNS = [
    ('javascript', re.compile(r'javascript', re.I)),
    ('json',       re.compile(r'json',       re.I)),
    ('xlsx',       re.compile(r'xlsx',       re.I)),
    ('docx',       re.compile(r'docx',       re.I)),
    ('pptx',       re.compile(r'pptx',       re.I)),
    ('html',       re.compile(r'html?',      re.I)),
    ('jpeg',       re.compile(r'jpe?g',      re.I)),
    ('7zip',       re.compile(r'7z(?:ip)?',  re.I)),
    ('pdf',        re.compile(r'pdf',        re.I)),
    ('zip',        re.compile(r'\bzip\b',    re.I)),
    ('apk',        re.compile(r'apk',        re.I)),
    ('exe',        re.compile(r'exe',        re.I)),
    ('dll',        re.compile(r'dll',        re.I)),
    ('mp3',        re.compile(r'mp3',        re.I)),
    ('mp4',        re.compile(r'mp4',        re.I)),
    ('png',        re.compile(r'png',        re.I)),
    ('gif',        re.compile(r'gif',        re.I)),
    ('bmp',        re.compile(r'bmp',        re.I)),
    ('tif',        re.compile(r'tiff?',      re.I)),
    ('csv',        re.compile(r'csv',        re.I)),
    ('xml',        re.compile(r'xml',        re.I)),
    ('svg',        re.compile(r'svg',        re.I)),
    ('eps',        re.compile(r'eps',        re.I)),
    ('swf',        re.compile(r'swf',        re.I)),
    ('elf',        re.compile(r'elf',        re.I)),
    ('rtf',        re.compile(r'rtf',        re.I)),
    ('bin',        re.compile(r'\bbin\b',    re.I)),
    ('css',        re.compile(r'css',        re.I)),
    ('txt',        re.compile(r'txt',        re.I)),
    ('tar',        re.compile(r'tar',        re.I)),
]


def infer_file_type_from_name(filename):
    """
    Infer the file type from a fragment filename using regex patterns.
    Returns uppercase type string or None.
    """
    for type_name, pattern in _TYPE_PATTERNS:
        if pattern.search(filename):
            return type_name.upper()
    return None


def infer_source_id(filename):
    """
    Extracts source file id from fragment name.
    E.g. "0001-pdf_trimmed_frag3_pdf.bin" → "0001-pdf"
         "fragment_10.bin" → None
    """
    match = re.match(r'^(\d{4})-(.+?)_trimmed', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_fragments(feature_matrix, n_samples=None, file_types=None, sub_cluster=False):
    """
    Clusters features using DBSCAN with K-Means fallback.

    If file_types are provided, fragments are grouped by type.
    If sub_cluster is True, further sub-clustering is done within each type
    to separate different original files of the same type.
    """
    if file_types is not None and len(set(t for t in file_types if t)) > 1:
        return _cluster_by_type_group(feature_matrix, file_types, sub_cluster)

    return _cluster_single_group(feature_matrix, n_samples)


def _cluster_single_group(feature_matrix, n_samples=None):
    """Cluster a single homogeneous group of fragments."""
    if feature_matrix.shape[0] < 2:
        return np.array([0] * feature_matrix.shape[0]), 'Single fragment'

    # Adaptive eps based on sample count
    target_cos_sim = 0.90
    target_cos_distance = 1.0 - target_cos_sim
    eps_euclidean = np.sqrt(2 * target_cos_distance)

    print(f"\n  Attempting DBSCAN (cosine threshold >= {target_cos_sim}) ...")
    dbscan = DBSCAN(
        eps=eps_euclidean,
        min_samples=2,
        metric='euclidean',
        algorithm='auto',
        n_jobs=-1,
    )
    labels = dbscan.fit_predict(feature_matrix)

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int((labels == -1).sum())

    if n_clusters <= 1 or (n_noise > 0.3 * len(labels)):
        print(f"    -> DBSCAN: {n_clusters} cluster(s), {n_noise} noise "
              f"({100.0 * n_noise / len(labels):.1f}%)")
        print("    -> Falling back to K-Means ...")

        if n_samples is None:
            n_samples = len(labels)
        labels, method_used = _kmeans_with_silhouette(feature_matrix, n_samples)
    else:
        method_used = 'DBSCAN'

    return labels, method_used


def _kmeans_with_silhouette(feature_matrix, n_samples):
    """
    Run K-Means with silhouette-guided k selection.
    Tries k in [2 .. min(sqrt(n), 20)] and picks the k with best silhouette.
    """
    max_k = min(int(np.sqrt(n_samples)), 20, n_samples - 1)
    if max_k < 2:
        max_k = 2

    best_score, best_k, best_labels = -1, 2, None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbls = km.fit_predict(feature_matrix)
        if len(set(lbls)) < 2:
            continue
        score = silhouette_score(feature_matrix, lbls, metric='cosine',
                                 sample_size=min(5000, len(lbls)))
        if score > best_score:
            best_score, best_k, best_labels = score, k, lbls

    if best_labels is None:
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        best_labels = km.fit_predict(feature_matrix)
        best_k = 2

    print(f"    -> K-Means selected k={best_k} (silhouette={best_score:.4f})")
    return best_labels, f'K-Means (k={best_k}, silhouette={best_score:.4f})'


def _cluster_by_type_group(feature_matrix, file_types, sub_cluster=False):
    """
    Clustering by file type.

    Default: One cluster per file type (all PDFs together, all HTMLs together).
    With sub_cluster=True: Further sub-clusters within each type to separate
    different original files of the same type.
    """
    print("\nGrouping fragments by inferred file type ...")

    # Map type → indices
    type_groups = {}
    for i, ft in enumerate(file_types):
        key = ft if ft else 'UNKNOWN'
        type_groups.setdefault(key, []).append(i)

    for ft, indices in sorted(type_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {ft}: {len(indices)} fragments")

    # Global label counter ensures unique labels across groups
    global_labels = np.full(len(file_types), -1, dtype=int)
    next_label = 0
    methods = []

    if not sub_cluster:
        # Simple: one cluster per file type
        for ft, indices in sorted(type_groups.items()):
            for idx in indices:
                global_labels[idx] = next_label
            methods.append(ft)
            next_label += 1

        method_str = 'Type-Grouped (one cluster per type: ' + ', '.join(methods) + ')'
        return global_labels, method_str

    # Sub-clustering within each type group
    print("\nSub-clustering within each type group ...\n")

    for ft, indices in sorted(type_groups.items()):
        indices = np.array(indices)
        sub_matrix = feature_matrix[indices]
        print(f"  [{ft}] {len(indices)} fragments:")

        if len(indices) == 1:
            global_labels[indices[0]] = next_label
            next_label += 1
            methods.append(f"  {ft}: single fragment")
            continue

        sub_labels, method = _cluster_single_group(sub_matrix, n_samples=len(indices))

        # Remap local labels to global
        for local_lbl in set(sub_labels):
            if local_lbl == -1:
                continue
            mask = sub_labels == local_lbl
            global_labels[indices[mask]] = next_label
            next_label += 1

        # Noise points get their own individual clusters
        noise_mask = sub_labels == -1
        for idx in indices[noise_mask]:
            global_labels[idx] = next_label
            next_label += 1

        methods.append(f"  {ft}: {method}")

    method_str = 'Type-Grouped + Sub-clustered: ' + '; '.join(methods)
    return global_labels, method_str


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_cluster_summary(labels, file_paths, method_used):
    # Group fragments by cluster
    cluster_fragments = {}
    for label, path in zip(labels, file_paths):
        cluster_fragments.setdefault(label, []).append(os.path.basename(path))

    print(f"\nClustering method used: {method_used}")
    print(f"Total clusters: {len(cluster_fragments)}")
    print("\n" + "="*70)

    for label in sorted(cluster_fragments.keys()):
        fragments = cluster_fragments[label]
        label_name = 'Noise / unclustered' if label == -1 else f'Cluster {label}'

        # Infer types for this cluster
        type_counts = {}
        for fname in fragments:
            ft = infer_file_type_from_name(fname)
            if ft:
                type_counts[ft] = type_counts.get(ft, 0) + 1
        type_summary = ', '.join(f'{t}:{c}' for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))

        print(f"\n  {label_name}: {len(fragments)} fragments  [{type_summary}]")
        print(f"  {'-'*60}")
        for i, fname in enumerate(sorted(fragments), 1):
            print(f"    {i:3d}. {fname}")

    print("\n" + "="*70)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cluster mixed .bin fragment files from a directory.'
    )
    parser.add_argument(
        '--fragments-dir',
        type=str,
        default=None,
        help='Path to the fragments directory. '
             'Defaults to datasets/mix_fragments in the repository root.',
    )
    parser.add_argument(
        '--max-fragments',
        type=int,
        default=None,
        help='Maximum number of fragments to process (e.g. 10). Default: all.',
    )
    parser.add_argument(
        '--no-type-group',
        action='store_true',
        help='Disable file-type pre-grouping and cluster all fragments together.',
    )
    parser.add_argument(
        '--sub-cluster',
        action='store_true',
        help='Further sub-cluster within each type to separate different original files.',
    )
    return parser.parse_args()


def find_fragment_files(root_dir):
    """Find all .bin files recursively under root_dir."""
    fragment_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if filename.lower().endswith('.bin'):
                fragment_paths.append(os.path.join(dirpath, filename))
    return fragment_paths


def find_fragments_by_subdir(root_dir):
    """
    Find .bin fragments organized by subdirectory (one subdir per type).
    Returns dict: {subdir_name: [list of .bin paths]}
    """
    fragments_by_type = {}
    if not os.path.isdir(root_dir):
        return fragments_by_type

    for entry in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, entry)
        if not os.path.isdir(subdir):
            continue
        bins = []
        for fname in sorted(os.listdir(subdir)):
            if fname.lower().endswith('.bin'):
                bins.append(os.path.join(subdir, fname))
        if bins:
            fragments_by_type[entry] = bins

    return fragments_by_type


import random

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from sequence/ to project root

    # Point to Train folder
    default_fragments_dir = os.path.join(project_root, 'Train')
    fragments_dir = os.path.abspath(args.fragments_dir) if args.fragments_dir else default_fragments_dir
    results_dir = os.path.join(project_root, 'results')

    print('=== Fragment Clustering ===')
    print(f'Using fragments directory: {fragments_dir}')

    if not os.path.exists(fragments_dir):
        print(f"Error: Fragments directory not found at '{fragments_dir}'.")
        print('Please ensure the directory exists and contains the .bin fragments.')
        return

    # Results directory
    results_dir = os.path.join(project_root, 'results')

    # Find fragments organized by subdirectory
    fragments_by_subdir = find_fragments_by_subdir(fragments_dir)

    if fragments_by_subdir:
        # Has subdirectories — sample randomly from each
        n_per_type = args.max_fragments or 10  # default 10 per type

        fragment_paths = []
        print(f'\nSampling {n_per_type} random fragments from each type subdirectory:')
        print('-' * 60)

        for subdir_name, paths in sorted(fragments_by_subdir.items()):
            available = len(paths)
            pick_count = min(n_per_type, available)
            picked = random.sample(paths, pick_count)
            fragment_paths.extend(picked)
            print(f'  {subdir_name}: {pick_count} / {available} fragments')

        print('-' * 60)

        # Shuffle all fragments so they are mixed randomly
        random.shuffle(fragment_paths)
        print(f'\nShuffled {len(fragment_paths)} total fragments from '
              f'{len(fragments_by_subdir)} type directories.')
    else:
        # Flat directory — use old behavior
        fragment_paths = find_fragment_files(fragments_dir)
        if args.max_fragments and len(fragment_paths) > args.max_fragments:
            fragment_paths = random.sample(fragment_paths, args.max_fragments)
            random.shuffle(fragment_paths)
            print(f'Randomly selected and shuffled {args.max_fragments} fragments.')
        else:
            print(f'Found {len(fragment_paths)} .bin fragment files.')

    if not fragment_paths:
        print(f"Error: No '.bin' files found inside '{fragments_dir}'.")
        return

    # Print loaded fragments grouped by type
    loaded_by_type = {}
    for p in fragment_paths:
        ft = infer_file_type_from_name(os.path.basename(p)) or 'UNKNOWN'
        loaded_by_type.setdefault(ft, []).append(os.path.basename(p))

    print(f'\nLoaded fragments ({len(fragment_paths)} total):')
    print('-' * 60)
    for ft in sorted(loaded_by_type.keys()):
        names = loaded_by_type[ft]
        print(f'\n  {ft} ({len(names)} fragments):')
        for i, name in enumerate(names, 1):
            print(f'    {i:3d}. {name}')
    print('-' * 60)

    # Infer file types for type-aware clustering
    file_types = None
    if not args.no_type_group:
        file_types = [
            infer_file_type_from_name(os.path.basename(p)) for p in fragment_paths
        ]
        unique_types = set(t for t in file_types if t)
        if len(unique_types) > 1:
            print(f'\nDetected {len(unique_types)} distinct file types: '
                  f'{", ".join(sorted(unique_types))}')
        else:
            file_types = None  # single type — no benefit from grouping

    feature_matrix = prepare_feature_matrix(fragment_paths)
    labels, method_used = cluster_fragments(
        feature_matrix, n_samples=len(fragment_paths), file_types=file_types,
        sub_cluster=args.sub_cluster
    )

    # Build output mapping
    cluster_mapping = {}
    for label, file_path in zip(labels, fragment_paths):
        rel_path = os.path.normpath(os.path.relpath(file_path, fragments_dir))
        cluster_label = f'Cluster {label}' if label != -1 else 'Noise (Unclustered)'
        cluster_mapping[rel_path] = cluster_label

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = os.path.join(results_dir, 'clusters.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_mapping, f, indent=4)

    print_cluster_summary(labels, fragment_paths, method_used)
    print(f"\nSuccessfully clustered {len(fragment_paths)} fragments. "
          f"Results saved to {output_file}")


if __name__ == '__main__':
    main()