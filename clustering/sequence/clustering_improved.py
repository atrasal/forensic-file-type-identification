#!/usr/bin/env python3
"""
IMPROVED FORENSIC CLUSTERING - Enhanced Feature Extraction
Focus: Better discrimination of file types using distinctive features
"""

import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict
import struct

def purity_score(labels_true, labels_pred):
    """Calculate clustering purity score"""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels_true, labels_pred)
    w = linear_sum_assignment(-cm)
    return cm[w].sum() / len(labels_true)

# ============================================================================
# FEATURE EXTRACTION - IMPROVED FOR FILE TYPE DISCRIMINATION
# ============================================================================

def _extract_magic_bytes(data, n=32):
    """Extract first N bytes as magic signature"""
    magic = list(data[:min(n, len(data))]) + [0] * max(0, n - len(data))
    return np.array(magic[:n], dtype=np.uint8)


def _extract_magic_patterns(data):
    """Detect specific file type magic patterns"""
    features = np.zeros(20, dtype=np.uint8)
    
    # Check for various magic bytes
    if data.startswith(b'\x7FUELF'):  # 0
        features[0] = 1
    if data.startswith(b'BM'):  # 1
        features[1] = 1
    if data.startswith(b'%PDF'):  # 2
        features[2] = 1
    if data.startswith(b'PK\x03\x04'):  # 3 - ZIP/APK/XLSX
        features[3] = 1
    if data.startswith(b'\x89PNG'):  # 4
        features[4] = 1
    if data.startswith(b'\xFF\xD8\xFF'):  # 5 - JPEG
        features[5] = 1
    if data.startswith(b'GIF8'):  # 6
        features[6] = 1
    if data.startswith(b'RIFF'):  # 7
        features[7] = 1
    if data.startswith(b'ID3'):  # 8 - MP3 tag
        features[8] = 1
    if data.startswith(b'MZ'):  # 9 - PE/DLL
        features[9] = 1
    if data.startswith(b'BZh'):  # 10 - BZIP2
        features[10] = 1
    if data.startswith(b'\x1F\x8B'):  # 11 - GZIP
        features[11] = 1
    if data.startswith(b'Rar!'):  # 12 - RAR
        features[12] = 1
    if len(data) >= 4 and data[:4] == bytes([0xCA, 0xFE, 0xBA, 0xBE]):  # 13 - JAR/CLASS
        features[13] = 1
    if len(data) >= 4 and data[:4] == bytes([0x37, 0x7a, 0xBC, 0xAF]):  # 14 - 7zip (37 7a BC AF)
        features[14] = 1
    if data.startswith(b'\x78\x9C'):  # 15 - zlib
        features[15] = 1
    if len(data) >= 4 and data[:4] == bytes([0xFE, 0xED, 0xFA, 0xCE]):  # 16 - Mach-O
        features[16] = 1
    # Additional text signatures
    if b'<?xml' in data or b'<!DOCTYPE' in data:  # 17
        features[17] = 1
    if b'<html' in data or b'<body' in data:  # 18 - HTML
        features[18] = 1
    if b'@charset' in data or b'body {' in data:  # 19 - CSS
        features[19] = 1
    
    return features


def _extract_entropy_blocks(data, block_size=32):
    """Shannon entropy of byte blocks"""
    n_blocks = 8
    entropies = []
    for i in range(n_blocks):
        start = i * len(data) // n_blocks
        end = (i + 1) * len(data) // n_blocks
        block = data[start:end]
        if len(block) > 0:
            hist = np.bincount(bytearray(block), minlength=256)
            probs = hist / len(block)
            entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
            entropies.append(entropy)
    return np.array(entropies, dtype=np.float32)


def _extract_byte_statistics(data):
    """Statistical features of byte distribution"""
    data_arr = np.frombuffer(data, dtype=np.uint8)
    stats = np.array([
        np.mean(data_arr),           # mean byte value
        np.std(data_arr),            # std dev
        np.min(data_arr),            # min
        np.max(data_arr),            # max
        np.percentile(data_arr, 25), # Q1
        np.percentile(data_arr, 50), # median
        np.percentile(data_arr, 75), # Q3
        np.sum(data_arr == 0) / len(data_arr),  # null byte ratio
        np.sum(data_arr >= 128) / len(data_arr),  # high bytes ratio
    ], dtype=np.float32)
    return stats


def _extract_text_likelihood(data):
    """Likelihood of being ASCII/text content"""
    text_chars = sum(1 for b in data if 32 <= b < 127 or b in (9, 10, 13))
    text_ratio = text_chars / max(len(data), 1)
    
    # Additional text indicators
    common_text = sum(1 for b in data if chr(b) in ' etion')
    common_ratio = common_text / max(len(data), 1)
    
    return np.array([text_ratio, common_ratio], dtype=np.float32)


def _extract_null_runs(data):
    """Detect patterns of null bytes (common in binary files)"""
    null_runs = []
    current_run = 0
    for b in data:
        if b == 0:
            current_run += 1
        else:
            if current_run > 0:
                null_runs.append(current_run)
            current_run = 0
    
    if null_runs:
        return np.array([
            len(null_runs),      # number of runs
            np.mean(null_runs),  # avg run length
            np.max(null_runs),   # max run length
        ], dtype=np.float32)
    else:
        return np.array([0, 0, 0], dtype=np.float32)


def _extract_ascii_patterns(data):
    """Look for ASCII/text structural patterns"""
    patterns = {
        b'<?xml': 1,  # XML
        b'<!DOCTYPE': 2,  # HTML
        b'<head>': 3,  # HTML
        b'<html': 4,  # HTML
        b'{': 5,  # JSON/CSS
        b'}': 5,  # JSON/CSS
        b'@charset': 6,  # CSS
        b'body': 7,  # CSS
    }
    
    features = np.zeros(len(patterns), dtype=np.uint8)
    for i, (pattern, _) in enumerate(patterns.items()):
        if pattern in data:
            features[i] = 1
    return features


def _extract_compression_indicators(data):
    """Detect compression signatures"""
    indicators = np.array([
        1 if data.startswith(b'\x1F\x8B') else 0,  # GZIP
        1 if data.startswith(b'BZh') else 0,       # BZIP2
        1 if data.startswith(b'\x78\x9C') else 0,  # zlib
        1 if data.startswith(b'\x37\x7a\xBC\xAF') else 0,  # 7zip
        1 if data.startswith(b'Rar!') else 0,      # RAR
        1 if b'.jar' in data or b'.class' in data else 0,  # Java
    ], dtype=np.uint8)
    return indicators


def extract_features(file_path, max_size=4096):
    """Extract comprehensive feature vector"""
    with open(file_path, 'rb') as f:
        data = f.read(max_size)
    
    if not data:
        return np.zeros(100)  # fallback
    
    features = []
    
    # 1. Magic bytes (first 32 bytes)
    features.append(_extract_magic_bytes(data, 32))
    
    # 2. Magic pattern detection (16 features)
    features.append(_extract_magic_patterns(data))
    
    # 3. Entropy blocks (8 blocks)
    features.append(_extract_entropy_blocks(data))
    
    # 4. Byte statistics (9 features)
    features.append(_extract_byte_statistics(data))
    
    # 5. Text likelihood (2 features)
    features.append(_extract_text_likelihood(data))
    
    # 6. Null byte patterns (3 features)
    features.append(_extract_null_runs(data))
    
    # 7. ASCII patterns (8 features)
    features.append(_extract_ascii_patterns(data))
    
    # 8. Compression indicators (6 features)
    features.append(_extract_compression_indicators(data))
    
    return np.concatenate(features).flatten().astype(np.float32)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*100)
    print("FORENSIC CLUSTERING PIPELINE - IMPROVED FEATURES")
    print("="*100)
    
    # Discover types
    train_dir = Path("Train")
    type_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    file_types = [d.name for d in type_dirs]
    
    print(f"\n[STEP 1] DISCOVER FILE TYPES")
    print("-" * 100)
    print(f"Auto-discovered {len(file_types)} file types: {', '.join(file_types)}\n")
    
    # Load fragments
    print(f"[STEP 2] LOAD FRAGMENTS")
    print("-" * 100)
    
    fragments_per_type = 10
    all_fragments = []
    all_labels = []
    all_filenames = []
    type_to_idx = {ft: i for i, ft in enumerate(file_types)}
    
    for file_type in file_types:
        type_dir = train_dir / file_type
        files = sorted(list(type_dir.glob("*.bin")))[:fragments_per_type]
        print(f"  {file_type}: {len(files)} fragments")
        
        for fpath in files:
            all_fragments.append(fpath)
            all_labels.append(type_to_idx[file_type])
            all_filenames.append(fpath.name)
    
    print(f"\n  Total fragments: {len(all_fragments)}\n")
    
    # Extract features
    print(f"[STEP 3] FEATURE EXTRACTION (IMPROVED)")
    print("-" * 100)
    print("  Extracting enhanced features …")
    
    features_list = []
    for i, frag in enumerate(all_fragments):
        feat = extract_features(str(frag))
        features_list.append(feat)
        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{len(all_fragments)} done")
    
    X = np.array(features_list)
    print(f"  Raw feature matrix: {X.shape}")
    
    # Scale and reduce
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=40)  # Reduced to 40 for new features
    X_reduced = pca.fit_transform(X_scaled)
    variance = np.sum(pca.explained_variance_ratio_)
    print(f"  After StandardScaler + PCA(40): {X_reduced.shape}  [{variance*100:.1f}% variance]\n")
    
    # Type-grouped clustering
    print(f"[STEP 4] TYPE-GROUPED CLUSTERING")
    print("-" * 100)
    
    all_cluster_labels = np.zeros(len(all_fragments), dtype=int)
    cluster_counter = 0
    
    for file_type in file_types:
        # Extract type identifier from directory name
        # e.g., "7zipFragments" -> "7zip", "apkFragments" -> "apk"
        if file_type.endswith('Fragments'):
            type_id = file_type[:-9]  # Remove 'Fragments'
        else:
            type_id = file_type
        
        # Match by file content in the name, not just extension
        # Look for patterns like "_7zip.bin", "_apk.bin", etc.
        mask = np.array([f'_{type_id}.' in all_filenames[i].lower() 
                         for i in range(len(all_filenames))])
        type_indices = np.where(mask)[0]
        
        # If no match with underscore pattern, try without underscore
        if len(type_indices) == 0:
            mask = np.array([type_id in all_filenames[i].lower() 
                             for i in range(len(all_filenames))])
            type_indices = np.where(mask)[0]
        
        if len(type_indices) == 0:
            print(f"\n  [{file_type}] 0 fragment(s) — skipping")
            continue
        
        X_type = X_reduced[type_indices]
        print(f"\n  [{file_type}] {len(type_indices)} fragment(s) — sub-clustering …")
        
        if len(type_indices) <= 2:
            # Too small for clustering
            for idx in type_indices:
                all_cluster_labels[idx] = cluster_counter
            cluster_counter += 1
        else:
            # Try DBSCAN first with adaptive eps
            dbscan = DBSCAN(eps=0.8, min_samples=2)
            sub_labels = dbscan.fit_predict(X_type)
            n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
            
            if n_clusters == 0:
                print(f"      DBSCAN gave 0 clusters, {len(type_indices)} noise — falling back to K-Means …")
                # Select k adaptively: sqrt(n) or minimum 2
                k = max(2, int(np.sqrt(len(type_indices))))
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                sub_labels = kmeans.fit_predict(X_type)
                sil = silhouette_score(X_type, sub_labels)
                print(f"      K-Means selected k={k} (silhouette={sil:.4f})")
            else:
                # Filter out noise points for silhouette calculation
                noise_mask = sub_labels != -1
                if np.any(noise_mask) and len(np.unique(sub_labels[noise_mask])) > 1:
                    sil = silhouette_score(X_type[noise_mask], sub_labels[noise_mask])
                else:
                    sil = 0.0
                print(f"      DBSCAN found {n_clusters} cluster(s) + {np.sum(sub_labels == -1)} noise (silhouette={sil:.4f})")
            
            # Map sub-labels to global labels
            unique_sub = np.unique(sub_labels)
            for unique_lbl in unique_sub:
                mask_sub = (sub_labels == unique_lbl)
                for idx in type_indices[mask_sub]:
                    all_cluster_labels[idx] = cluster_counter
                cluster_counter += 1
    
    print(f"\n  Total clusters created: {cluster_counter}\n")
    
    # Compute metrics
    print(f"[STEP 5] CLUSTERING QUALITY METRICS")
    print("-" * 100)
    
    true_labels = np.array(all_labels)
    
    purity = purity_score(true_labels, all_cluster_labels)
    ari = adjusted_rand_score(true_labels, all_cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, all_cluster_labels)
    
    print(f"  Purity : {purity:.4f}  (1.0 = perfect)")
    print(f"  ARI    : {ari:.4f}   (closer to 1.0 = perfect)")
    print(f"  NMI    : {nmi:.4f}   (closer to 1.0 = perfect)\n")
    
    # Print detailed cluster assignments (subset)
    print(f"[STEP 6] CLUSTER ASSIGNMENTS (First 10 clusters)\n")
    print("-" * 100)
    
    cluster_to_items = defaultdict(list)
    for idx, (frag, true_label, pred_label) in enumerate(zip(all_filenames, true_labels, all_cluster_labels)):
        cluster_to_items[pred_label].append((frag, true_label))
    
    for cluster_id in sorted(cluster_to_items.keys())[:10]:
        items = cluster_to_items[cluster_id]
        print(f"\nCLUSTER {cluster_id}  ({len(items)} fragments)")
        
        # Count types in cluster
        type_counts = defaultdict(int)
        for frag, true_label in items:
            type_counts[file_types[true_label]] += 1
        
        type_str = ', '.join([f"{ft}:{ct}" for ft, ct in sorted(type_counts.items())])
        print(f"  Composition: {type_str}")
        
        # Show first 3 items
        for frag, true_label in items[:3]:
            match = "✓" if file_types[true_label].lower() in frag.lower() else "✗"
            print(f"    {match} {frag}")
    
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"Total fragments: {len(all_fragments)}")
    print(f"Total clusters: {len(cluster_to_items)}")
    print(f"Purity: {purity:.2%}")
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
