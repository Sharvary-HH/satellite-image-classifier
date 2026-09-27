"""Step 0: inspect the raw tile dataset before choosing a model.

Usage: python scripts/inspect_data.py [DATA_DIR]   (default: data/raw)

Reports file counts and formats, folder structure (labelled or not),
image size / band count / dtype / value range, and any embedded metadata
(PNG text chunks, EXIF, GeoTIFF tags) that could carry georeferencing.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
files = sorted(p for p in data_dir.rglob("*") if p.is_file() and not p.name.startswith("."))

print(f"Data dir: {data_dir.resolve()}")
print(f"Total files: {len(files)}")
print(f"Extensions: {dict(Counter(p.suffix.lower() for p in files))}")

# Folder structure: number of files per parent folder (per-class folders => labelled).
print("\nFiles per folder:")
for folder, n in sorted(Counter(str(p.parent.relative_to(data_dir)) for p in files).items()):
    print(f"  {folder:35s} {n}")

# Open every image and collect size, mode (band layout), dtype and value range.
sizes, modes, dtypes, metadata_keys = Counter(), Counter(), Counter(), Counter()
global_min, global_max, unreadable = 255, 0, []
for p in files:
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        continue
    try:
        with Image.open(p) as img:
            sizes[img.size] += 1
            modes[img.mode] += 1
            metadata_keys.update(img.info.keys())  # PNG text chunks, dpi, etc.
            if img.getexif():
                metadata_keys["EXIF"] += 1
            arr = np.asarray(img)
    except Exception as e:
        unreadable.append((p.name, str(e)))
        continue
    dtypes[str(arr.dtype)] += 1
    global_min, global_max = min(global_min, int(arr.min())), max(global_max, int(arr.max()))

print(f"\nImage sizes (w, h): {dict(sizes)}")
print(f"PIL modes (band layout): {dict(modes)}")
print(f"Array dtypes: {dict(dtypes)}")
print(f"Pixel value range across all images: [{global_min}, {global_max}]")
print(f"Metadata keys found (count of files): {dict(metadata_keys) or 'none'}")
print(f"Unreadable files: {unreadable or 'none'}")
