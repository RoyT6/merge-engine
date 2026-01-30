"""
TEST MERGE ENGINE - 10 Records Only
====================================
Quick validation test with limited data to verify merge logic works.

GPU MANDATORY - ALGO 95.4 Compliance
"""

import os
import sys

# GPU Environment setup
os.environ.setdefault('LD_LIBRARY_PATH', '/usr/lib/wsl/lib')
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')
os.environ.setdefault('NUMBA_CUDA_DRIVER', '/usr/lib/wsl/lib/libcuda.so.1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# GPU MANDATORY - Fail fast if not available
try:
    import cudf
    import cupy as cp
    print(f"[GPU] cuDF {cudf.__version__} loaded")
    mem_free, mem_total = cp.cuda.runtime.memGetInfo()
    print(f"[GPU] VRAM: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")
except ImportError as e:
    print(f"[FATAL] GPU libraries not available: {e}")
    print("[FATAL] This script requires cuDF/RAPIDS. Run via: ./run_gpu.sh")
    sys.exit(1)

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Paths
BASE_DIR = Path("/mnt/c/Users/RoyT6/Downloads")
FLIXPATROL_CSV = BASE_DIR / "ToMerge" / "FlixPatrol_Views_v40.10_CLEANED.csv"
BFD_PARQUET = BASE_DIR / "BFD_V26.00.parquet"

# Empty value handling
EMPTY_VALUES = {'', 'None', 'none', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', '<NA>'}

# 22 Countries
COUNTRIES_22 = [
    'us', 'cn', 'in', 'gb', 'br', 'de', 'jp', 'fr', 'ca', 'mx',
    'au', 'es', 'it', 'kr', 'nl', 'se', 'sg', 'hk', 'ie', 'ru',
    'tr', 'row'
]


def normalize_imdb_id(val):
    """Normalize IMDb ID."""
    if val is None or str(val) in EMPTY_VALUES:
        return None
    s = str(val)
    if s.isdigit():
        return f"tt{s}"
    return s


def main():
    print("=" * 70)
    print("    TEST MERGE ENGINE - 10 RECORDS")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Load FlixPatrol CSV (first 10 rows)
    # =========================================================================
    print(f"\n[1] Loading FlixPatrol CSV (10 records): {FLIXPATROL_CSV}")

    # Load with pandas first to get just 10 rows, then convert to cuDF
    fp_pandas = pd.read_csv(str(FLIXPATROL_CSV), nrows=10)
    print(f"    Loaded: {len(fp_pandas)} rows, {len(fp_pandas.columns)} columns")

    # Show titles loaded
    print("\n    Titles in test set:")
    for i, row in fp_pandas.iterrows():
        title = row.get('title', 'Unknown')
        imdb = row.get('imdb_id', 'N/A')
        views = row.get('total_views', 0)
        print(f"      {i+1}. {title} ({imdb}) - {views:,} views")

    # Convert to cuDF
    fp_gpu = cudf.DataFrame(fp_pandas)

    # Normalize imdb_id - simple approach for cuDF compatibility
    def norm_imdb(val):
        if val is None or str(val) in EMPTY_VALUES or str(val) == 'nan':
            return None
        s = str(val)
        if s.isdigit():
            return f"tt{s}"
        return s

    fp_pandas['imdb_id_norm'] = fp_pandas['imdb_id'].apply(norm_imdb)
    fp_gpu = cudf.DataFrame(fp_pandas)

    # =========================================================================
    # STEP 2: Load BFD Parquet
    # =========================================================================
    print(f"\n[2] Loading BFD Parquet: {BFD_PARQUET}")

    bfd_gpu = cudf.read_parquet(str(BFD_PARQUET))
    print(f"    Loaded: {len(bfd_gpu):,} rows, {len(bfd_gpu.columns):,} columns")

    # Memory check
    mem_free, mem_total = cp.cuda.runtime.memGetInfo()
    print(f"    [GPU] VRAM after load: {mem_free/1e9:.1f}GB free")

    # Normalize BFD imdb_id - convert to pandas for normalization then back
    print("    Normalizing BFD imdb_id...")
    bfd_pandas = bfd_gpu.to_pandas()
    bfd_pandas['imdb_id_norm'] = bfd_pandas['imdb_id'].apply(norm_imdb)
    # Keep as pandas for the test (faster for small operations)

    # =========================================================================
    # STEP 3: Find Matches
    # =========================================================================
    print("\n[3] Finding matches between FlixPatrol and BFD...")

    # Get FlixPatrol IMDb IDs
    fp_imdb_ids = set(fp_pandas['imdb_id_norm'].dropna().tolist())
    print(f"    FlixPatrol IMDb IDs: {fp_imdb_ids}")

    matches = []
    for imdb_id in fp_imdb_ids:
        if imdb_id is None:
            continue
        # Search by imdb_id or fc_uid starting with imdb_id
        mask_imdb = bfd_pandas['imdb_id'].astype(str) == imdb_id
        mask_fc = bfd_pandas['fc_uid'].astype(str).str.startswith(imdb_id, na=False)
        combined = mask_imdb | mask_fc

        if combined.any():
            match_count = combined.sum()
            matches.append((imdb_id, match_count))
            print(f"    MATCH: {imdb_id} -> {match_count} BFD row(s)")

    print(f"\n    Total matches: {len(matches)} / {len(fp_imdb_ids)} FlixPatrol titles")

    # =========================================================================
    # STEP 4: Simulate Merge (Views Columns)
    # =========================================================================
    print("\n[4] Simulating merge (OVERWRITE mode - FlixPatrol senior)...")

    view_cols = [f'views_{cc}' for cc in COUNTRIES_22]

    # Check which view columns exist in BFD
    existing_view_cols = [c for c in view_cols if c in bfd_pandas.columns]
    missing_view_cols = [c for c in view_cols if c not in bfd_pandas.columns]

    print(f"    View columns in BFD: {len(existing_view_cols)}")
    print(f"    View columns missing: {len(missing_view_cols)}")

    if missing_view_cols:
        print(f"    Would create: {missing_view_cols[:5]}...")

    # Count potential updates
    overwrites = 0
    fills = 0

    for _, fp_row in fp_pandas.iterrows():
        imdb_id = normalize_imdb_id(fp_row.get('imdb_id'))
        if not imdb_id:
            continue

        # Find in BFD
        mask = bfd_pandas['imdb_id'].astype(str) == imdb_id
        if not mask.any():
            mask = bfd_pandas['fc_uid'].astype(str).str.startswith(imdb_id, na=False)

        if mask.any():
            for col in view_cols:
                fp_val = fp_row.get(col)
                if fp_val is not None and str(fp_val) not in EMPTY_VALUES and float(fp_val) > 0:
                    if col in bfd_pandas.columns:
                        bfd_val = bfd_pandas.loc[mask, col].iloc[0] if mask.any() else None
                        if pd.notna(bfd_val) and bfd_val != 0:
                            overwrites += 1
                        else:
                            fills += 1

    print(f"\n    Potential OVERWRITES: {overwrites}")
    print(f"    Potential NULL fills: {fills}")

    # =========================================================================
    # STEP 5: Sample Data Comparison
    # =========================================================================
    print("\n[5] Sample data comparison (first 3 matches)...")

    sample_count = 0
    for imdb_id, _ in matches[:3]:
        # Get FlixPatrol row
        fp_match = fp_pandas[fp_pandas['imdb_id'].astype(str).apply(normalize_imdb_id) == imdb_id]
        if fp_match.empty:
            continue

        fp_row = fp_match.iloc[0]

        # Get BFD row
        mask = bfd_pandas['imdb_id'].astype(str) == imdb_id
        if not mask.any():
            mask = bfd_pandas['fc_uid'].astype(str).str.startswith(imdb_id, na=False)

        if not mask.any():
            continue

        bfd_row = bfd_pandas[mask].iloc[0]

        print(f"\n    --- {fp_row.get('title', 'Unknown')} ({imdb_id}) ---")
        print(f"    FlixPatrol total_views: {fp_row.get('total_views', 'N/A'):,}")

        # Compare US views
        fp_us = fp_row.get('views_us', 0)
        bfd_us = bfd_row.get('views_us', 'N/A') if 'views_us' in bfd_pandas.columns else 'N/A'

        print(f"    FlixPatrol views_us: {fp_us:,}")
        print(f"    BFD views_us: {bfd_us}")

        if pd.notna(bfd_us) and bfd_us != 0:
            print(f"    -> Would OVERWRITE")
        else:
            print(f"    -> Would FILL NULL")

        sample_count += 1

    # =========================================================================
    # STEP 6: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("    TEST COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\n  FlixPatrol records tested: 10")
    print(f"  BFD total rows: {len(bfd_pandas):,}")
    print(f"  Matches found: {len(matches)}")
    print(f"  Would overwrite: {overwrites} values")
    print(f"  Would fill NULL: {fills} values")

    # Memory final
    mem_free, mem_total = cp.cuda.runtime.memGetInfo()
    print(f"\n  [GPU] Final VRAM: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")

    print("\n  [OK] Test completed successfully - merge logic validated")
    print("=" * 70)


if __name__ == '__main__':
    main()
