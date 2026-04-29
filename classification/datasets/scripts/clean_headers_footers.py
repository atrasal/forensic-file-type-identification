"""
Header/Footer Cleaner — scans fragment files, detects file-type signatures,
and removes fragments that contain headers or footers.

Usage:
  python datasets/scripts/clean_headers_footers.py --input Train
  python datasets/scripts/clean_headers_footers.py --input Train --dry-run   # preview only
"""

import os
import sys
import csv
import argparse


# ========== Signature database ==========
# Each entry: (name, header_bytes_to_check, footer_bytes_to_check)
# header is checked in the first 32 bytes, footer in the last 32 bytes
HEADER_SIGNATURES = [
    # Documents
    (b'%PDF',),                  # PDF
    (b'{\\rtf',),                # RTF
    (b'PK\x03\x04',),           # ZIP/DOCX/XLSX/APK/JAR
    
    # Images
    (b'\xff\xd8\xff',),          # JPEG
    (b'\x89PNG\r\n\x1a\n',),    # PNG
    (b'GIF8',),                  # GIF
    (b'BM',),                    # BMP
    (b'II\x2a\x00',),           # TIFF (little-endian)
    (b'MM\x00\x2a',),           # TIFF (big-endian)
    
    # Audio/Video
    (b'\xff\xfb',),              # MP3 frame
    (b'\xff\xf3',),              # MP3 frame (MPEG2)
    (b'\xff\xf2',),              # MP3 frame (MPEG2.5)
    (b'ID3',),                   # MP3 ID3 tag
    (b'RIFF',),                  # WAV/AVI
    (b'fLaC',),                  # FLAC
    (b'\x1a\x45\xdf\xa3',),     # MKV/WebM
    (b'OggS',),                  # OGG
    
    # Archives
    (b'7z\xbc\xaf\x27\x1c',),   # 7-Zip
    (b'Rar!\x1a\x07',),         # RAR
    (b'\x1f\x8b',),             # GZIP
    
    # Executables
    (b'MZ',),                    # PE/EXE
    (b'\x7fELF',),              # ELF
    (b'\xfe\xed\xfa',),         # Mach-O
    (b'\xca\xfe\xba\xbe',),     # Java class / Mach-O fat
    (b'dex\n',),                 # Android DEX
    
    # Text-based (check in first 32 bytes)
    (b'<!DOCTYPE',),             # HTML
    (b'<html',),                 # HTML
    (b'<?xml',),                 # XML
    (b'<svg',),                  # SVG
]

FOOTER_SIGNATURES = [
    (b'%%EOF',),                 # PDF
    (b'\xff\xd9',),              # JPEG
    (b'IEND',),                  # PNG
    (b'\x00\x3b',),              # GIF
    (b'PK\x05\x06',),           # ZIP end of central directory
    (b'</html>',),              # HTML
    (b'</svg>',),               # SVG
]

# Text-based formats: check for structural markers
TEXT_HEADERS = [
    b'<!DOCTYPE', b'<html', b'<?xml', b'<svg',
    b'/*', b'var ', b'function ', b'const ', b'let ',
    b'import ', b'from ', b'export ',
    b'{"', b'[{',
]

TEXT_FOOTERS = [
    b'</html>', b'</svg>', b'</body>',
]


def read_fragment(path):
    """Read a fragment file and return raw bytes. Handles both hex-encoded and binary."""
    with open(path, 'rb') as f:
        raw = f.read()
    
    # Try to decode as hex text
    try:
        text = raw.decode('ascii').strip()
        # If it looks like hex (all hex chars), decode it
        if all(c in '0123456789abcdefABCDEF' for c in text):
            return bytes.fromhex(text)
    except:
        pass
    
    return raw  # raw binary


def has_header(data, check_bytes=32):
    """Check if fragment data starts with a known file header signature."""
    prefix = data[:check_bytes]
    
    for sig_tuple in HEADER_SIGNATURES:
        sig = sig_tuple[0]
        if prefix[:len(sig)] == sig:
            return True, sig
    
    # Check text-based headers
    for sig in TEXT_HEADERS:
        if sig in prefix:
            return True, sig
    
    return False, None


def has_footer(data, check_bytes=32):
    """Check if fragment data ends with a known file footer signature."""
    suffix = data[-check_bytes:]
    
    for sig_tuple in FOOTER_SIGNATURES:
        sig = sig_tuple[0]
        if sig in suffix:
            return True, sig
    
    # Check text-based footers
    for sig in TEXT_FOOTERS:
        if sig in suffix:
            return True, sig
    
    return False, None


def clean_directory(input_dir, dry_run=False):
    """
    Scan all fragment subfolders, detect and remove fragments with headers/footers.
    Also updates the CSV label files.
    """
    total_scanned = 0
    total_removed = 0
    
    for folder_name in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Gather all fragment files (non-CSV, non-hidden)
        frag_files = sorted([
            f for f in os.listdir(folder_path) 
            if not f.endswith('.csv') and not f.startswith('.')
        ])
        
        header_hits = []
        footer_hits = []
        to_remove = set()
        
        for fname in frag_files:
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            
            total_scanned += 1
            data = read_fragment(fpath)
            
            h_found, h_sig = has_header(data)
            f_found, f_sig = has_footer(data)
            
            if h_found:
                header_hits.append((fname, h_sig))
                to_remove.add(fname)
            if f_found:
                footer_hits.append((fname, f_sig))
                to_remove.add(fname)
        
        removed_count = len(to_remove)
        remaining = len(frag_files) - removed_count
        total_removed += removed_count
        
        status = "✅ CLEAN" if removed_count == 0 else f"🗑️  removing {removed_count}"
        print(f"{folder_name:25s} | scanned {len(frag_files):6d} | "
              f"headers: {len(header_hits):4d} | footers: {len(footer_hits):4d} | "
              f"removing: {removed_count:4d} | remaining: {remaining:5d} | {status}")
        
        if not dry_run and to_remove:
            # Delete the fragment files
            for fname in to_remove:
                os.remove(os.path.join(folder_path, fname))
            
            # Update CSV label file if it exists
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            for csv_file in csv_files:
                csv_path = os.path.join(folder_path, csv_file)
                updated_rows = []
                header_row = None
                
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    header_row = next(reader)
                    for row in reader:
                        # The first column is the fragment filename
                        if row and row[0] not in to_remove:
                            updated_rows.append(row)
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header_row)
                    writer.writerows(updated_rows)
    
    print(f"\n{'='*80}")
    print(f"📊 Total scanned: {total_scanned}")
    print(f"🗑️  Total removed: {total_removed}")
    print(f"✅ Total remaining: {total_scanned - total_removed}")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN — no files were actually deleted. Run without --dry-run to delete.")


def main():
    parser = argparse.ArgumentParser(
        description="Detect and remove fragments containing file headers/footers"
    )
    parser.add_argument('--input', default='Train',
                        help='Directory with fragment subfolders (default: Train)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what would be removed without actually deleting')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"❌ Directory not found: {args.input}")
        sys.exit(1)
    
    mode = "DRY RUN (preview only)" if args.dry_run else "LIVE (will delete files)"
    print(f"🔍 Header/Footer Cleaner")
    print(f"   Input:  {args.input}")
    print(f"   Mode:   {mode}")
    print()
    
    clean_directory(args.input, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
