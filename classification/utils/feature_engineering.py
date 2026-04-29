"""
Feature engineering for file fragment classification.

Extracts hand-crafted features from raw byte sequences for use with
tree-based models (Random Forest, XGBoost) that can't learn spatial
patterns like CNNs.

All features are derived purely from raw byte content — no metadata,
headers, or footers are used.

Features extracted per fragment:
  - Byte frequency histogram (256 features)
  - Shannon entropy (1 feature)
  - Byte-pair (bigram) top-N frequencies (20 features)
  - Statistical features: mean, std, skewness, kurtosis, median, etc. (10 features)
  - Block entropy over sub-blocks (16 features)
  - Longest byte run and zero-byte run (2 features)
  - Chi-squared randomness test statistic (1 feature)
  - Compression ratio estimate via zlib (1 feature)
  - Trigram top-N frequencies (10 features)

Total: ~317 features per fragment
"""


import numpy as np
import zlib
from scipy import stats


def byte_frequency_histogram(fragment):
    """
    Count frequency of each byte value (0-255).
    Returns a normalized 256-element vector.
    """
    hist = np.bincount(fragment.astype(int), minlength=256)
    return hist / len(fragment)  # Normalize to proportions


def shannon_entropy(fragment):
    """
    Calculate Shannon entropy of byte distribution.
    Higher entropy = more random/compressed data.
    Lower entropy = more structured data (text, etc.)
    """
    hist = np.bincount(fragment.astype(int), minlength=256)
    probs = hist / len(fragment)
    probs = probs[probs > 0]  # Remove zero entries
    return -np.sum(probs * np.log2(probs))


def bigram_frequencies(fragment, top_n=20):
    """
    Count frequency of byte pairs (bigrams).
    Returns the top N most common bigram frequencies.
    """
    frag_int = fragment.astype(int)
    # Create bigram indices: byte1 * 256 + byte2
    bigrams = frag_int[:-1] * 256 + frag_int[1:]
    hist = np.bincount(bigrams, minlength=65536)
    # Return top N frequencies, normalized
    top_freqs = np.sort(hist)[::-1][:top_n]
    return top_freqs / len(bigrams)


def statistical_features(fragment):
    """
    Basic statistical features of the byte sequence.
    """
    frag = fragment.astype(float)
    features = [
        np.mean(frag),                          # Mean byte value
        np.std(frag),                           # Standard deviation
        float(stats.skew(frag)),                # Skewness
        float(stats.kurtosis(frag)),            # Kurtosis
        np.median(frag),                        # Median
        float(np.percentile(frag, 25)),         # Q1
        float(np.percentile(frag, 75)),         # Q3
        float(np.max(frag) - np.min(frag)),     # Range
        float(np.sum(frag == 0)) / len(frag),   # Zero byte ratio
        float(np.sum((frag >= 32) & (frag <= 126))) / len(frag),  # ASCII printable ratio
    ]
    return np.nan_to_num(np.array(features), nan=0.0)


def extract_features(fragment):
    """
    Extract all engineered features from a single fragment.

    Args:
        fragment: numpy array of byte values (0-255 or 0.0-1.0 normalized)

    Returns:
        1D numpy array of all features concatenated
    """
    # If normalized (0-1), scale back to 0-255 for feature extraction
    if fragment.max() <= 1.0:
        frag = (fragment * 255).astype(int)
    else:
        frag = fragment.astype(int)

    frag = np.clip(frag, 0, 255)

    features = np.concatenate([
        byte_frequency_histogram(frag),   # 256 features
        [shannon_entropy(frag)],           # 1 feature
        bigram_frequencies(frag, top_n=20),# 20 features
        statistical_features(frag),        # 10 features
        block_entropy(frag),               # 16 features
        longest_runs(frag),                # 2 features
        [chi_squared_test(frag)],          # 1 feature
        [compression_ratio(frag)],         # 1 feature
        trigram_frequencies(frag, top_n=10),# 10 features
    ])

    return features


def extract_features_batch(X):
    """
    Extract features for a batch of fragments.

    Args:
        X: numpy array of shape (n_samples, fragment_size)

    Returns:
        numpy array of shape (n_samples, n_features)
    """
    print(f"🔧 Extracting engineered features from {X.shape[0]} fragments...")
    features = np.array([extract_features(x) for x in X], dtype=np.float32)
    print(f"   → {features.shape[1]} features per fragment")
    return features


# Feature names for interpretability
def get_feature_names():
    """Get human-readable names for all features."""
    names = []
    names += [f"byte_freq_{i}" for i in range(256)]
    names += ["entropy"]
    names += [f"bigram_top_{i}" for i in range(20)]
    names += ["mean", "std", "skewness", "kurtosis", "median",
              "q1", "q3", "range", "zero_ratio", "ascii_printable_ratio"]
    names += [f"block_entropy_{i}" for i in range(16)]
    names += ["longest_byte_run", "longest_zero_run"]
    names += ["chi_squared"]
    names += ["compression_ratio"]
    names += [f"trigram_top_{i}" for i in range(10)]
    return names


# ========== New Forensic Features ==========

def block_entropy(fragment, n_blocks=16):
    """
    Calculate Shannon entropy over sub-blocks of the fragment.
    Helps distinguish files with uniform vs. varying internal structure.
    Returns n_blocks entropy values.
    """
    block_size = len(fragment) // n_blocks
    entropies = []
    for i in range(n_blocks):
        block = fragment[i * block_size:(i + 1) * block_size]
        hist = np.bincount(block.astype(int), minlength=256)
        probs = hist / len(block)
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log2(probs)))
    return np.array(entropies)


def longest_runs(fragment):
    """
    Find the longest consecutive run of the same byte value,
    and the longest run of zero bytes.
    Useful for detecting padding, sparse data, or structured regions.
    """
    frag = fragment.astype(int)
    max_run = 1
    max_zero_run = 0
    current_run = 1
    current_zero_run = 1 if frag[0] == 0 else 0

    for i in range(1, len(frag)):
        if frag[i] == frag[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

        if frag[i] == 0:
            if frag[i - 1] == 0:
                current_zero_run += 1
            else:
                current_zero_run = 1
            max_zero_run = max(max_zero_run, current_zero_run)
        else:
            current_zero_run = 0

    # Normalize by fragment length
    return np.array([max_run / len(frag), max_zero_run / len(frag)])


def chi_squared_test(fragment):
    """
    Chi-squared test for byte distribution uniformity.
    High values = non-uniform (structured data like text).
    Low values = uniform (compressed/encrypted, looks random).
    Helps distinguish compressed archives from other types.
    """
    observed = np.bincount(fragment.astype(int), minlength=256).astype(float)
    expected = np.full(256, len(fragment) / 256.0)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    # Normalize to [0, 1] range approximately
    return chi2 / len(fragment)


def compression_ratio(fragment):
    """
    Estimate how compressible the data is using zlib.
    Already-compressed data won't compress further (ratio ≈ 1.0).
    Structured/text data compresses well (ratio << 1.0).
    """
    raw_bytes = bytes(fragment.astype(np.uint8))
    compressed = zlib.compress(raw_bytes, level=1)  # Fast compression
    return len(compressed) / len(raw_bytes)


def trigram_frequencies(fragment, top_n=10):
    """
    Count frequency of byte trigrams (3-grams).
    Returns the top N most common trigram frequencies.
    Uses np.unique instead of np.bincount to avoid allocating a 16M-element array.
    """
    frag_int = fragment.astype(int)
    # Create trigram indices: byte1 * 65536 + byte2 * 256 + byte3
    trigrams = frag_int[:-2] * 65536 + frag_int[1:-1] * 256 + frag_int[2:]
    _, counts = np.unique(trigrams, return_counts=True)
    # Sort descending and take top N
    top_counts = np.sort(counts)[::-1][:top_n]
    # Pad with zeros if fewer than top_n unique trigrams
    if len(top_counts) < top_n:
        top_counts = np.pad(top_counts, (0, top_n - len(top_counts)))
    return top_counts / len(trigrams)
