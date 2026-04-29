"""
Utility functions for file preprocessing and feature extraction.
"""
import numpy as np
from scipy import stats
import zlib

# ==================== FILE SIGNATURE DATABASE ====================
FILE_SIGNATURES = {
    # Images
    'jpg':  {'header': b'\xff\xd8\xff',          'footer': b'\xff\xd9',       'header_len': 20,  'footer_len': 2},
    'jpeg': {'header': b'\xff\xd8\xff',          'footer': b'\xff\xd9',       'header_len': 20,  'footer_len': 2},
    'png':  {'header': b'\x89PNG\r\n\x1a\n',     'footer': b'IEND',           'header_len': 33,  'footer_len': 12},
    'gif':  {'header': b'GIF8',                  'footer': b'\x00\x3b',       'header_len': 13,  'footer_len': 2},
    'bmp':  {'header': b'BM',                    'footer': None,              'header_len': 54,  'footer_len': 0},
    'tiff': {'header': b'II\x2a\x00',            'footer': None,              'header_len': 8,   'footer_len': 0},

    # Documents
    'pdf':  {'header': b'%PDF',                  'footer': b'%%EOF',          'header_len': 15,  'footer_len': 6},
    'docx': {'header': b'PK\x03\x04',            'footer': b'PK\x05\x06',    'header_len': 30,  'footer_len': 22},
    'xlsx': {'header': b'PK\x03\x04',            'footer': b'PK\x05\x06',    'header_len': 30,  'footer_len': 22},
    'pptx': {'header': b'PK\x03\x04',            'footer': b'PK\x05\x06',    'header_len': 30,  'footer_len': 22},
    'doc':  {'header': b'\xd0\xcf\x11\xe0',      'footer': None,              'header_len': 512, 'footer_len': 0},

    # Audio
    'mp3':  {'header': b'\xff\xfb',              'footer': None,              'header_len': 4,   'footer_len': 0},
    'wav':  {'header': b'RIFF',                  'footer': None,              'header_len': 44,  'footer_len': 0},
    'flac': {'header': b'fLaC',                  'footer': None,              'header_len': 4,   'footer_len': 0},

    # Video
    'mp4':  {'header': b'ftyp',                  'footer': None,              'header_len': 8,   'footer_len': 0},
    'avi':  {'header': b'RIFF',                  'footer': None,              'header_len': 12,  'footer_len': 0},
    'mkv':  {'header': b'\x1a\x45\xdf\xa3',      'footer': None,              'header_len': 4,   'footer_len': 0},

    # Archives
    'zip':  {'header': b'PK\x03\x04',            'footer': b'PK\x05\x06',    'header_len': 30,  'footer_len': 22},
    'rar':  {'header': b'Rar!\x1a\x07',          'footer': None,              'header_len': 7,   'footer_len': 0},
    'gz':   {'header': b'\x1f\x8b',              'footer': None,              'header_len': 10,  'footer_len': 0},

    # Executables
    'exe':  {'header': b'MZ',                    'footer': None,              'header_len': 64,  'footer_len': 0},
    'elf':  {'header': b'\x7fELF',               'footer': None,              'header_len': 64,  'footer_len': 0},
}


# ==================== HEADER/FOOTER DETECTION & REMOVAL ====================

def detect_header_footer(data, file_ext):
    """
    Detect whether the file data contains known header/footer signatures.
    Returns dict with detection results.
    """
    ext = file_ext.lower().lstrip('.')
    result = {
        'has_header': False,
        'has_footer': False,
        'header_len': 0,
        'footer_len': 0,
        'file_type': ext,
        'detected_type': 'Unknown'
    }

    sig = FILE_SIGNATURES.get(ext)
    if not sig:
        # Try to match by content even if extension unknown
        for file_type, sig_info in FILE_SIGNATURES.items():
            if sig_info['header'] and data[:len(sig_info['header'])] == sig_info['header']:
                result['file_type'] = file_type
                result['detected_type'] = file_type.upper()
                result['has_header'] = True
                result['header_len'] = sig_info['header_len']
                if sig_info['footer'] and sig_info['footer'] in data[-64:]:
                    result['has_footer'] = True
                    result['footer_len'] = sig_info['footer_len']
                return result
        return result

    # Check header
    if sig['header'] and data[:len(sig['header'])] == sig['header']:
        result['has_header'] = True
        result['header_len'] = sig['header_len']
        result['detected_type'] = ext.upper()

    # Check footer
    if sig['footer'] and sig['footer'] in data[-64:]:
        result['has_footer'] = True
        result['footer_len'] = sig['footer_len']

    if result['has_header'] or result['has_footer']:
        result['detected_type'] = ext.upper()

    return result


def clean_file_data(file_bytes, file_ext=''):
    """
    Clean file by removing detected headers/footers.
    Returns cleaned bytes, original stats, and cleaned stats.
    """
    original_size = len(file_bytes)
    detection = detect_header_footer(file_bytes, file_ext)
    
    # Strip header and footer
    header_len = detection['header_len'] if detection['has_header'] else 0
    footer_len = detection['footer_len'] if detection['has_footer'] else 0
    
    if footer_len > 0:
        cleaned_data = file_bytes[header_len:-footer_len]
    else:
        cleaned_data = file_bytes[header_len:]
    
    cleaned_size = len(cleaned_data)
    
    return cleaned_data, detection, {
        'original_size': original_size,
        'cleaned_size': cleaned_size,
        'header_removed': header_len,
        'footer_removed': footer_len,
        'bytes_removed': header_len + footer_len,
        'removal_percentage': (header_len + footer_len) / original_size * 100 if original_size > 0 else 0
    }


def create_fragments(data, chunk_size=4096, num_fragments=5):
    """
    Create multiple fragments from cleaned data.
    Samples from beginning, middle, and end for better diversity.
    """
    if len(data) < chunk_size:
        padded = np.pad(np.frombuffer(data, dtype=np.uint8), 
                       (0, chunk_size - len(data)), mode='constant')
        return [padded]
    
    fragments = []
    positions = np.linspace(0, max(0, len(data) - chunk_size), num_fragments, dtype=int)
    
    for pos in positions:
        fragment = np.frombuffer(data[pos:pos + chunk_size], dtype=np.uint8)
        if len(fragment) < chunk_size:
            fragment = np.pad(fragment, (0, chunk_size - len(fragment)), mode='constant')
        fragments.append(fragment)
    
    return fragments


# ==================== FEATURE EXTRACTION FUNCTIONS ====================

def byte_frequency_histogram(fragment):
    """Count frequency of each byte value (0-255)."""
    hist = np.bincount(fragment.astype(int), minlength=256)
    return hist / len(fragment)


def shannon_entropy(fragment):
    """Calculate Shannon entropy of byte distribution."""
    hist = np.bincount(fragment.astype(int), minlength=256)
    probs = hist / len(fragment)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0


def bigram_frequencies(fragment, top_n=20):
    """Count frequency of byte pairs (bigrams)."""
    frag_int = fragment.astype(int)
    bigrams = frag_int[:-1] * 256 + frag_int[1:]
    hist = np.bincount(bigrams, minlength=65536)
    top_freqs = np.sort(hist)[::-1][:top_n]
    result = top_freqs / len(bigrams) if len(bigrams) > 0 else np.zeros(top_n)
    if len(result) < top_n:
        result = np.pad(result, (0, top_n - len(result)))
    return result


def statistical_features(fragment):
    """Extract basic statistical features."""
    frag = fragment.astype(float)
    features = [
        np.mean(frag),
        np.std(frag),
        float(stats.skew(frag)),
        float(stats.kurtosis(frag)),
        np.median(frag),
        float(np.percentile(frag, 25)),
        float(np.percentile(frag, 75)),
        float(np.max(frag) - np.min(frag)),
        float(np.sum(frag == 0)) / len(frag),
        float(np.sum((frag >= 32) & (frag <= 126))) / len(frag),
    ]
    return np.nan_to_num(np.array(features), nan=0.0)


def block_entropy(fragment, n_blocks=16):
    """Calculate Shannon entropy over sub-blocks."""
    block_size = max(1, len(fragment) // n_blocks)
    entropies = []
    for i in range(n_blocks):
        block = fragment[i * block_size:(i + 1) * block_size]
        if len(block) == 0:
            entropies.append(0)
            continue
        hist = np.bincount(block.astype(int), minlength=256)
        probs = hist / len(block)
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0)
    return np.array(entropies)


def longest_runs(fragment):
    """Find longest consecutive run of same byte and zero bytes."""
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
    return np.array([max_run / len(frag), max_zero_run / len(frag)])


def chi_squared_test(fragment):
    """Chi-squared test for byte distribution uniformity."""
    observed = np.bincount(fragment.astype(int), minlength=256).astype(float)
    expected = np.full(256, len(fragment) / 256.0)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    return chi2 / len(fragment)


def compression_ratio(fragment):
    """Estimate compressibility using zlib."""
    raw_bytes = bytes(fragment.astype(np.uint8))
    compressed = zlib.compress(raw_bytes, level=1)
    return len(compressed) / len(raw_bytes)


def trigram_frequencies(fragment, top_n=10):
    """Count frequency of byte trigrams (3-grams)."""
    frag_int = fragment.astype(int)
    if len(frag_int) < 3:
        return np.zeros(top_n)
    trigrams = frag_int[:-2] * 65536 + frag_int[1:-1] * 256 + frag_int[2:]
    _, counts = np.unique(trigrams, return_counts=True)
    top_counts = np.sort(counts)[::-1][:top_n]
    if len(top_counts) < top_n:
        top_counts = np.pad(top_counts, (0, top_n - len(top_counts)))
    return top_counts / len(trigrams) if len(trigrams) > 0 else np.zeros(top_n)


def extract_features(file_bytes):
    """Extract all engineered features from file bytes."""
    fragment = np.frombuffer(file_bytes, dtype=np.uint8)
    if len(fragment) == 0:
        return np.zeros(317)
    
    # Pad or truncate to 4096 bytes
    if len(fragment) < 4096:
        fragment = np.pad(fragment, (0, 4096 - len(fragment)), mode='constant')
    else:
        fragment = fragment[:4096]
    
    # Normalize to [0, 1] to match training pipeline
    fragment = fragment.astype(np.float32) / 255.0
    
    features = np.concatenate([
        byte_frequency_histogram(fragment),
        [shannon_entropy(fragment)],
        bigram_frequencies(fragment, top_n=20),
        statistical_features(fragment),
        block_entropy(fragment),
        longest_runs(fragment),
        [chi_squared_test(fragment)],
        [compression_ratio(fragment)],
        trigram_frequencies(fragment, top_n=10),
    ])
    
    return features.astype(np.float32)
