"""
File Fragmenter — converts whole files into fixed-size binary fragments.

Features:
  - Detects known file headers and footers using a signature database
  - Reports whether headers/footers are present
  - Optionally strips headers and footers before fragmenting
  - Saves fragments as .hex text files with a mapping CSV

Usage:
  python datasets/scripts/fragmenter.py --input datasets/raw --output datasets/fragments
  python datasets/scripts/fragmenter.py --input datasets/raw --output datasets/fragments --no-strip
  python datasets/scripts/fragmenter.py --input datasets/raw --output datasets/fragments --chunk-size 512
"""

import os
import sys
import csv
import argparse
import random

# ========== File Signature Database ==========
# Maps file type to (header_bytes, footer_bytes, header_length, footer_length)
# None means no known signature for that part
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

    # Audio
    'mp3':  {'header': b'\xff\xfb',              'footer': None,              'header_len': 4,   'footer_len': 0},
    'wav':  {'header': b'RIFF',                  'footer': None,              'header_len': 44,  'footer_len': 0},
    'flac': {'header': b'fLaC',                  'footer': None,              'header_len': 4,   'footer_len': 0},

    # Video
    'mp4':  {'header': None,                     'footer': None,              'header_len': 8,   'footer_len': 0},
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

# Also detect ID3 tags in MP3 files
MP3_ID3_HEADER = b'ID3'


def detect_header_footer(data, file_ext):
    """
    Detect whether the file data contains known header/footer signatures.
    
    Returns:
        dict with keys: has_header, has_footer, header_len, footer_len, file_type
    """
    ext = file_ext.lower().lstrip('.')
    result = {
        'has_header': False,
        'has_footer': False,
        'header_len': 0,
        'footer_len': 0,
        'file_type': ext,
    }

    sig = FILE_SIGNATURES.get(ext)
    if not sig:
        return result

    # Check header
    if sig['header'] and data[:len(sig['header'])] == sig['header']:
        result['has_header'] = True
        result['header_len'] = sig['header_len']

    # Special case: MP3 with ID3 tag
    if ext in ('mp3',) and data[:3] == MP3_ID3_HEADER:
        result['has_header'] = True
        # ID3v2 header size is in bytes 6-9 (syncsafe integer)
        if len(data) > 10:
            size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
            result['header_len'] = 10 + size

    # Special case: MP4 ftyp box detection
    if ext == 'mp4' and len(data) > 8:
        if data[4:8] == b'ftyp':
            result['has_header'] = True
            box_size = int.from_bytes(data[0:4], 'big')
            result['header_len'] = box_size

    # Check footer
    if sig['footer'] and sig['footer'] in data[-64:]:
        result['has_footer'] = True
        result['footer_len'] = sig['footer_len']

    return result


def fragment_file(data, chunk_size, strip_header=0, strip_footer=0):
    """Split binary data into fixed-size chunks after stripping header/footer."""
    if strip_footer > 0:
        data = data[strip_header:-strip_footer]
    else:
        data = data[strip_header:]

    if len(data) < chunk_size:
        return []

    fragments = []
    for i in range(0, len(data) - chunk_size + 1, chunk_size):
        fragments.append(data[i:i + chunk_size])
    return fragments


def save_hex_fragment(fragment_bytes, path):
    """Save a binary fragment as a hex string text file."""
    with open(path, 'w') as f:
        hex_string = ''.join(format(b, '02x') for b in fragment_bytes)
        f.write(hex_string)


def process_dataset(input_dir, output_dir, chunk_size=1024, strip=True,
                    manual_header=None, manual_footer=None):
    """
    Process all files in input_dir/<filetype>/ subfolders into fragments.
    
    Args:
        input_dir:      Path to raw files (subfolders = file types)
        output_dir:     Path to save fragments and mapping CSV
        chunk_size:     Size of each fragment in bytes
        strip:          Whether to strip detected headers/footers
        manual_header:  Manual header bytes to strip (overrides auto-detect)
        manual_footer:  Manual footer bytes to strip (overrides auto-detect)
    """
    os.makedirs(output_dir, exist_ok=True)
    mapping_path = os.path.join(output_dir, "fragment_mapping.csv")

    fragment_counter = 1
    mapping_rows = []
    stats = {}

    for filetype in sorted(os.listdir(input_dir)):
        type_path = os.path.join(input_dir, filetype)
        if not os.path.isdir(type_path):
            continue

        type_count = 0
        header_found = 0
        footer_found = 0

        for filename in sorted(os.listdir(type_path)):
            file_path = os.path.join(type_path, filename)
            if not os.path.isfile(file_path):
                continue

            with open(file_path, 'rb') as f:
                data = f.read()

            # Detect header/footer
            file_ext = os.path.splitext(filename)[1]
            detection = detect_header_footer(data, file_ext if file_ext else filetype)

            if detection['has_header']:
                header_found += 1
            if detection['has_footer']:
                footer_found += 1

            # Determine strip amounts
            if strip:
                h = manual_header if manual_header is not None else (detection['header_len'] if detection['has_header'] else 0)
                ft = manual_footer if manual_footer is not None else (detection['footer_len'] if detection['has_footer'] else 0)
            else:
                h = manual_header if manual_header is not None else 0
                ft = manual_footer if manual_footer is not None else 0

            fragments = fragment_file(data, chunk_size, strip_header=h, strip_footer=ft)

            for frag in fragments:
                out_path = os.path.join(output_dir, f"{fragment_counter}.hex")
                save_hex_fragment(frag, out_path)
                mapping_rows.append([fragment_counter, filetype])
                fragment_counter += 1

            type_count += len(fragments)

        stats[filetype] = {
            'fragments': type_count,
            'headers_detected': header_found,
            'footers_detected': footer_found,
        }
        print(f"  {filetype}: {type_count} fragments "
              f"(headers: {header_found}, footers: {footer_found})")

    # Save mapping CSV
    with open(mapping_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fragment_id', 'file_type'])
        writer.writerows(mapping_rows)

    total = fragment_counter - 1
    print(f"\n✅ Fragmentation complete: {total} total fragments")
    print(f"📄 Mapping CSV saved to: {mapping_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fragment files into fixed-size hex chunks for ML training"
    )
    parser.add_argument('--input', default='datasets/raw',
                        help='Input directory with file-type subfolders (default: datasets/raw)')
    parser.add_argument('--output', default='datasets/fragments',
                        help='Output directory for fragments (default: datasets/fragments)')
    parser.add_argument('--chunk-size', type=int, default=1024,
                        help='Fragment size in bytes (default: 1024)')
    parser.add_argument('--no-strip', action='store_true',
                        help='Do NOT strip detected headers/footers')
    parser.add_argument('--header-bytes', type=int, default=None,
                        help='Manual: strip this many header bytes (overrides auto-detect)')
    parser.add_argument('--footer-bytes', type=int, default=None,
                        help='Manual: strip this many footer bytes (overrides auto-detect)')

    args = parser.parse_args()

    print(f"🔧 Fragmenter Configuration:")
    print(f"   Input:      {args.input}")
    print(f"   Output:     {args.output}")
    print(f"   Chunk size: {args.chunk_size} bytes")
    print(f"   Strip H/F:  {'No' if args.no_strip else 'Yes (auto-detect)'}")
    if args.header_bytes is not None:
        print(f"   Manual header strip: {args.header_bytes} bytes")
    if args.footer_bytes is not None:
        print(f"   Manual footer strip: {args.footer_bytes} bytes")
    print()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        strip=not args.no_strip,
        manual_header=args.header_bytes,
        manual_footer=args.footer_bytes,
    )


if __name__ == "__main__":
    main()
