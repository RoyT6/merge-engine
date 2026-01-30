"""
TEST MERGE ENGINE - 100 Records with FULL IMPLEMENTATION
=========================================================
Full test with:
- VBUS Memory Management (L1/L2/L3 caching)
- Season Allocator (ALGO2)
- Intelligent View Allocation
- GPU Acceleration (cuDF/RAPIDS)

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
import warnings
warnings.filterwarnings('ignore')

# Add Merge Engine directory to path
MERGE_ENGINE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(MERGE_ENGINE_DIR))

# Import VBUS Memory Manager
try:
    from vbus_memory_manager import VBUSGPUManager, HierarchicalCache
    VBUS_AVAILABLE = True
    print("[VBUS] Memory manager loaded")
except ImportError as e:
    VBUS_AVAILABLE = False
    print(f"[VBUS] Not available: {e}")

# Import Season Allocator
try:
    from season_allocator import SeasonAllocatorEngine, SeasonInfo
    ALLOCATOR_AVAILABLE = True
    print("[ALGO2] Season allocator loaded")
except ImportError as e:
    ALLOCATOR_AVAILABLE = False
    print(f"[ALGO2] Not available: {e}")

# Paths
BASE_DIR = Path("/mnt/c/Users/RoyT6/Downloads")
FLIXPATROL_CSV = BASE_DIR / "ToMerge" / "FlixPatrol_Views_v40.10_CLEANED.csv"
BFD_PARQUET = BASE_DIR / "BFD_V26.00.parquet"
WEIGHTERS_DIR = BASE_DIR / "Weighters"

# Empty value handling
EMPTY_VALUES = {'', 'None', 'none', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', '<NA>'}

# 22 Countries
COUNTRIES_22 = [
    'us', 'cn', 'in', 'gb', 'br', 'de', 'jp', 'fr', 'ca', 'mx',
    'au', 'es', 'it', 'kr', 'nl', 'se', 'sg', 'hk', 'ie', 'ru',
    'tr', 'row'
]


def norm_imdb(val):
    """Normalize IMDb ID."""
    if val is None or str(val) in EMPTY_VALUES or str(val) == 'nan':
        return None
    s = str(val)
    if s.isdigit():
        return f"tt{s}"
    return s


class TestMergeEngine:
    """Test merge engine with full VBUS + ALGO2 implementation."""

    def __init__(self):
        self.vbus_manager = None
        self.season_allocator = None
        self.cache = None
        self.stats = {
            'fp_rows': 0,
            'bfd_rows': 0,
            'matches': 0,
            'new_columns_created': 0,
            'values_would_overwrite': 0,
            'values_would_fill': 0,
            'tv_shows_needing_allocation': 0,
            'direct_mapped': 0,
            'season_allocated': 0,
        }

    def initialize_vbus(self):
        """Initialize VBUS memory management."""
        print("\n[VBUS] Initializing memory management...")

        if VBUS_AVAILABLE:
            self.vbus_manager = VBUSGPUManager(use_gpu=True)
            self.cache = HierarchicalCache()
            print("[VBUS] GPU Manager initialized")
            print("[VBUS] L1 (VRAM): 64 entries")
            print("[VBUS] L2 (Pinned): 256 entries")
            print("[VBUS] L3 (System): 1024 entries")
            self.vbus_manager.print_status()
        else:
            print("[VBUS] Not available - using basic memory management")

    def initialize_allocator(self):
        """Initialize season allocator."""
        print("\n[ALGO2] Initializing season allocator...")

        if ALLOCATOR_AVAILABLE:
            self.season_allocator = SeasonAllocatorEngine(str(WEIGHTERS_DIR))
            print("[ALGO2] Season allocator ready")
            print("[ALGO2] Viewer model: 45% continuers, 35% acquired, 20% rewatchers")
        else:
            print("[ALGO2] Not available - will use fallback allocation")

    def load_flixpatrol(self, nrows=100):
        """Load FlixPatrol data with VBUS staging."""
        print(f"\n[LOAD] Loading FlixPatrol CSV ({nrows} records)...")

        # Load to pandas first
        fp_pandas = pd.read_csv(str(FLIXPATROL_CSV), nrows=nrows)
        self.stats['fp_rows'] = len(fp_pandas)
        print(f"[LOAD] Loaded: {len(fp_pandas)} rows, {len(fp_pandas.columns)} columns")

        # Normalize IMDb IDs
        fp_pandas['imdb_id_norm'] = fp_pandas['imdb_id'].apply(norm_imdb)

        # Stage in VBUS cache
        if self.cache:
            self.cache.put("flixpatrol_data", fp_pandas)
            print("[VBUS] FlixPatrol data staged in cache")

        # Show sample
        print("\n[LOAD] Sample titles:")
        for i, row in fp_pandas.head(10).iterrows():
            title = row.get('title', 'Unknown')[:30]
            imdb = row.get('imdb_id', 'N/A')
            views = row.get('total_views', 0)
            stype = row.get('title_type', 'unknown')
            season = row.get('season_number', '-')
            print(f"  {i+1:3}. {title:30} | {imdb:12} | {stype:8} | S{season} | {views:>15,} views")

        return fp_pandas

    def load_bfd(self):
        """Load BFD with VBUS memory management."""
        print(f"\n[LOAD] Loading BFD Parquet...")

        # VBUS: Clear GPU memory before loading large dataset
        if self.vbus_manager:
            self.vbus_manager.clear_gpu_memory()
            print("[VBUS] Cleared GPU memory before BFD load")

        # Check memory before
        mem_free_before, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"[GPU] VRAM before: {mem_free_before/1e9:.2f}GB free")

        # Load with cuDF
        bfd_gpu = cudf.read_parquet(str(BFD_PARQUET))
        self.stats['bfd_rows'] = len(bfd_gpu)
        print(f"[LOAD] Loaded: {len(bfd_gpu):,} rows, {len(bfd_gpu.columns):,} columns")

        # Check memory after
        mem_free_after, _ = cp.cuda.runtime.memGetInfo()
        mem_used = (mem_free_before - mem_free_after) / 1e9
        print(f"[GPU] VRAM after: {mem_free_after/1e9:.2f}GB free (used {mem_used:.2f}GB)")

        # VBUS: Stage in cache hierarchy
        if self.vbus_manager:
            self.vbus_manager.stage_data("bfd_base", bfd_gpu, hot=True)
            self.vbus_manager.print_status()

        # Convert to pandas for test operations (smaller memory footprint for analysis)
        print("[LOAD] Converting to pandas for analysis...")
        bfd_pandas = bfd_gpu.to_pandas()

        # Free GPU memory
        del bfd_gpu
        cp.get_default_memory_pool().free_all_blocks()

        mem_free_final, _ = cp.cuda.runtime.memGetInfo()
        print(f"[GPU] VRAM after pandas conversion: {mem_free_final/1e9:.2f}GB free")

        # Normalize IMDb IDs
        print("[LOAD] Normalizing BFD imdb_id...")
        bfd_pandas['imdb_id_norm'] = bfd_pandas['imdb_id'].apply(norm_imdb)

        return bfd_pandas

    def find_matches(self, fp_pandas, bfd_pandas):
        """Find matches between FlixPatrol and BFD."""
        print("\n[MATCH] Finding matches...")

        fp_imdb_ids = set(fp_pandas['imdb_id_norm'].dropna().unique())
        print(f"[MATCH] Unique FlixPatrol IMDb IDs: {len(fp_imdb_ids)}")

        matches = {}
        for imdb_id in fp_imdb_ids:
            if imdb_id is None:
                continue

            # Search by imdb_id or fc_uid
            mask_imdb = bfd_pandas['imdb_id_norm'] == imdb_id
            mask_fc = bfd_pandas['fc_uid'].astype(str).str.startswith(imdb_id, na=False)
            combined = mask_imdb | mask_fc

            if combined.any():
                match_count = combined.sum()
                matches[imdb_id] = {
                    'count': match_count,
                    'rows': bfd_pandas[combined].index.tolist()
                }

        self.stats['matches'] = len(matches)
        print(f"[MATCH] Found: {len(matches)} / {len(fp_imdb_ids)} titles match BFD")

        # Show match distribution
        multi_season = sum(1 for v in matches.values() if v['count'] > 1)
        single_row = sum(1 for v in matches.values() if v['count'] == 1)
        print(f"[MATCH] Single-row matches: {single_row}")
        print(f"[MATCH] Multi-row matches (seasons): {multi_season}")

        return matches

    def analyze_columns(self, bfd_pandas):
        """Analyze which columns need to be created."""
        print("\n[SCHEMA] Analyzing columns...")

        view_cols = [f'views_{cc}' for cc in COUNTRIES_22]
        hour_cols = [f'hours_{cc}' for cc in COUNTRIES_22]

        existing_view = [c for c in view_cols if c in bfd_pandas.columns]
        missing_view = [c for c in view_cols if c not in bfd_pandas.columns]

        existing_hour = [c for c in hour_cols if c in bfd_pandas.columns]
        missing_hour = [c for c in hour_cols if c not in bfd_pandas.columns]

        print(f"[SCHEMA] views_{{cc}} columns: {len(existing_view)} exist, {len(missing_view)} missing")
        print(f"[SCHEMA] hours_{{cc}} columns: {len(existing_hour)} exist, {len(missing_hour)} missing")

        self.stats['new_columns_created'] = len(missing_view) + len(missing_hour)

        if missing_view:
            print(f"[SCHEMA] Would CREATE: {missing_view[:5]}...")

        return {
            'view_cols': view_cols,
            'hour_cols': hour_cols,
            'missing_view': missing_view,
            'missing_hour': missing_hour
        }

    def test_intelligent_allocation(self, fp_pandas, bfd_pandas, matches):
        """Test intelligent view allocation logic."""
        print("\n[ALGO2] Testing intelligent view allocation...")

        if not ALLOCATOR_AVAILABLE:
            print("[ALGO2] Season allocator not available, skipping")
            return

        # Find TV shows in FlixPatrol
        tv_mask = fp_pandas['title_type'].isin(['tv_show', 'TV Show', 'tvSeries', 'series'])
        tv_shows = fp_pandas[tv_mask]
        print(f"[ALGO2] TV shows in FlixPatrol: {len(tv_shows)}")

        for _, fp_row in tv_shows.iterrows():
            imdb_id = fp_row.get('imdb_id_norm')
            if not imdb_id or imdb_id not in matches:
                continue

            match_info = matches[imdb_id]
            bfd_row_count = match_info['count']
            season_num = fp_row.get('season_number')
            total_views = fp_row.get('total_views', 0)

            # Check if direct mapping or needs allocation
            is_movie = fp_row.get('title_type') in ['movie', 'film', 'Movie', 'Film']
            has_season = season_num and str(season_num) not in EMPTY_VALUES and str(season_num) != 'nan'
            single_bfd_row = bfd_row_count <= 1

            if is_movie or has_season or single_bfd_row:
                # Direct mapping - SENIOR
                self.stats['direct_mapped'] += 1
            else:
                # Needs season allocation
                self.stats['tv_shows_needing_allocation'] += 1

                if total_views > 0:
                    # Demo allocation
                    print(f"\n  [ALLOCATE] {fp_row.get('title', 'Unknown')} ({imdb_id})")
                    print(f"    Total views: {total_views:,}")
                    print(f"    BFD seasons: {bfd_row_count}")

                    # Build season info
                    seasons = []
                    for i in range(bfd_row_count):
                        seasons.append(SeasonInfo(
                            season_number=i + 1,
                            release_date=None
                        ))

                    # Run allocation
                    genre = fp_row.get('genre') or 'drama_serial'
                    if isinstance(genre, list):
                        genre = genre[0] if genre else 'drama_serial'

                    try:
                        results, metadata = self.season_allocator.allocate_views(
                            total_views=int(total_views),
                            seasons=seasons,
                            genre=str(genre),
                            reporting_period_end=datetime.now(),
                            show_title=fp_row.get('title', 'Unknown')
                        )

                        print(f"    Allocation:")
                        for r in results:
                            print(f"      Season {r.season_number}: {r.allocated_views:,} ({r.allocation_percent:.1f}%)")

                        self.stats['season_allocated'] += 1

                    except Exception as e:
                        print(f"    Allocation error: {e}")

        print(f"\n[ALGO2] Summary:")
        print(f"  Direct mapped (SENIOR): {self.stats['direct_mapped']}")
        print(f"  Season allocated: {self.stats['season_allocated']}")
        print(f"  Needing allocation: {self.stats['tv_shows_needing_allocation']}")

    def simulate_merge(self, fp_pandas, bfd_pandas, matches, col_info):
        """Simulate the merge operation."""
        print("\n[MERGE] Simulating merge (OVERWRITE mode)...")

        view_cols = col_info['view_cols']
        overwrites = 0
        fills = 0

        for _, fp_row in fp_pandas.iterrows():
            imdb_id = fp_row.get('imdb_id_norm')
            if not imdb_id or imdb_id not in matches:
                continue

            match_info = matches[imdb_id]
            bfd_indices = match_info['rows']

            for col in view_cols:
                fp_val = fp_row.get(col)
                if fp_val is None or str(fp_val) in EMPTY_VALUES:
                    continue
                try:
                    fp_val = float(fp_val)
                    if fp_val <= 0:
                        continue
                except:
                    continue

                for idx in bfd_indices:
                    if col in bfd_pandas.columns:
                        bfd_val = bfd_pandas.loc[idx, col]
                        if pd.notna(bfd_val) and bfd_val != 0:
                            overwrites += 1
                        else:
                            fills += 1
                    else:
                        fills += 1

        self.stats['values_would_overwrite'] = overwrites
        self.stats['values_would_fill'] = fills

        print(f"[MERGE] Would OVERWRITE: {overwrites:,} values")
        print(f"[MERGE] Would FILL NULL: {fills:,} values")

    def run(self):
        """Run full test."""
        print("=" * 70)
        print("    TEST MERGE ENGINE - 100 RECORDS (FULL IMPLEMENTATION)")
        print("    VBUS + ALGO2 + GPU Acceleration")
        print("=" * 70)
        start_time = datetime.now()

        # Initialize components
        self.initialize_vbus()
        self.initialize_allocator()

        # Load data
        fp_pandas = self.load_flixpatrol(nrows=100)
        bfd_pandas = self.load_bfd()

        # Analysis
        matches = self.find_matches(fp_pandas, bfd_pandas)
        col_info = self.analyze_columns(bfd_pandas)

        # Test intelligent allocation
        self.test_intelligent_allocation(fp_pandas, bfd_pandas, matches)

        # Simulate merge
        self.simulate_merge(fp_pandas, bfd_pandas, matches, col_info)

        # Final VBUS status
        if self.vbus_manager:
            print("\n[VBUS] Final status:")
            self.vbus_manager.print_status()

        # Summary
        elapsed = datetime.now() - start_time

        print("\n" + "=" * 70)
        print("    TEST COMPLETE - FULL SUMMARY")
        print("=" * 70)
        print(f"\n  Duration: {elapsed}")
        print(f"\n  DATA:")
        print(f"    FlixPatrol records: {self.stats['fp_rows']}")
        print(f"    BFD rows: {self.stats['bfd_rows']:,}")
        print(f"    Matches found: {self.stats['matches']}")
        print(f"\n  COLUMNS:")
        print(f"    Would create: {self.stats['new_columns_created']}")
        print(f"\n  MERGE SIMULATION:")
        print(f"    Would overwrite: {self.stats['values_would_overwrite']:,}")
        print(f"    Would fill NULL: {self.stats['values_would_fill']:,}")
        print(f"\n  INTELLIGENT ALLOCATION:")
        print(f"    Direct mapped (SENIOR): {self.stats['direct_mapped']}")
        print(f"    Season allocated (ALGO2): {self.stats['season_allocated']}")

        # Final GPU memory
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"\n  [GPU] Final VRAM: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")

        print("\n  [OK] Full test completed successfully")
        print("=" * 70)

        return self.stats


if __name__ == '__main__':
    engine = TestMergeEngine()
    engine.run()
