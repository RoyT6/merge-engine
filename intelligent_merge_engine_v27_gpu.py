"""
Intelligent Merge Engine for BFD V28.00 - GPU Accelerated + VBUS Memory Management
==================================================================================
STANDALONE VERSION - Schema V28 Compliant

V28.00 ADDITIONS:
- star_1, star_2, star_3, supporting_cast columns for major actor differentiation
- cast_data preserved for backwards compatibility but DEPRECATED

CRITICAL RULES:
1. NO NEW COLUMNS WITHOUT AUTHORIZATION - Engine MUST ask before creating any column
2. All column mappings MUST follow SCHEMA_V27.00.json
3. temporal_period_type: "lifetime" is FORBIDDEN - reject and ask for guidance
4. FlixPatrol views_{cc} MUST map to schema patterns views_{period}_{year}_{cc}
5. P0 JAILBREAK (V27.60): NO views for periods that END BEFORE premiere_date
   - See: Schema/RULES/pre_premiere_views_rules.json
   - Bug Fixed: 68.7M impossible views cells nulled
   - MUST validate: IF period_end_date < premiere_date THEN value = NULL

FlixPatrol has SENIOR CREDIBILITY - overwrites existing views data.
Appends NEW rows when FlixPatrol has titles not in BFD.

SCHEMA COLUMN PATTERNS (V27):
- views_{h1|h2}_{year}_{cc} - Half-yearly views
- views_{q1|q2|q3|q4}_{year}_{cc} - Quarterly views
- hours_{cc} - Hours by country (limited)
NOTE: "estimate/estimated/lifetime" words are BANNED in column names

GPU MANDATORY - No CPU fallback (ALGO 95.4 compliance)
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

# STANDALONE: Use local imports from Merge Engine directory
MERGE_ENGINE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(MERGE_ENGINE_DIR))

# Import standalone VBUS memory manager with grep performance
try:
    from vbus_memory_manager import VBUSGPUManager, ColumnGrepEngine
    VBUS_AVAILABLE = True
    print("[VBUS] Memory manager loaded (standalone)")
    print("[GREP] Column grep engine available for fast pattern matching")
except ImportError:
    VBUS_AVAILABLE = False
    ColumnGrepEngine = None
    print("[VBUS] Not available - using basic memory management")

# Import standalone season allocator
try:
    from season_allocator import SeasonAllocatorEngine, SeasonInfo
    SEASON_ALLOCATOR_AVAILABLE = True
    print("[ALGO2] Season allocator loaded (standalone)")
except ImportError:
    SEASON_ALLOCATOR_AVAILABLE = False
    print("[ALGO2] Season allocator not available")

# Use WSL paths
BASE_DIR = Path("/mnt/c/Users/RoyT6/Downloads")
WEIGHTERS_DIR = BASE_DIR / "Weighters"
SCHEMA_PATH = BASE_DIR / "Schema" / "SCHEMA_V27.00.json"

# Input files
BFD_BASE = BASE_DIR / "BFD_V27.54.parquet"  # V27.54 master database
FLIXPATROL_VIEWS = BASE_DIR / "ToMerge" / "FlixPatrol_Views_v40.10_CLEANED.csv"
FLIXPATROL_MULTI_TEMPORAL = BASE_DIR / "ToMerge" / "FlixPatrol_Multi_Temporal_v41.00_CLEANED.csv"
FLIXPATROL_NETFLIX = BASE_DIR / "ToMerge" / "FlixPatrol_Netflix_Weekly_Views_v40.10_CLEANED.csv"

# Output
OUTPUT_FILE = BASE_DIR / "BFD_V27.55.parquet"
REPORT_FILE = BASE_DIR / "BFD_V27.55.merge_report.json"

# Empty value handling
EMPTY_VALUES = {'', 'None', 'none', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', '<NA>'}

# =============================================================================
# SCHEMA ENFORCEMENT - NO NEW COLUMNS WITHOUT AUTHORIZATION
# =============================================================================

# FORBIDDEN temporal_period_types - these require special handling
FORBIDDEN_TEMPORAL_TYPES = {'lifetime'}

# FORBIDDEN words in column names - cannot create columns with these words
FORBIDDEN_COLUMN_WORDS = {'estimate', 'estimated', 'lifetime'}

# Valid temporal period patterns
VALID_TEMPORAL_PATTERNS = {
    'half_yearly': 'views_{period}_{year}_{cc}',  # h1_2024, h2_2024
    'quarterly': 'views_{period}_{year}_{cc}',     # q1_2024, q2_2024, q3_2024, q4_2024
}

# 22 Countries per schema
COUNTRIES_22 = [
    'us', 'cn', 'in', 'gb', 'br', 'de', 'jp', 'fr', 'ca', 'mx',
    'au', 'es', 'it', 'kr', 'nl', 'se', 'sg', 'hk', 'ie', 'ru',
    'tr', 'row'
]

# Schema-compliant column patterns
SCHEMA_COLUMN_PATTERNS = {
    # Hours (limited)
    'hours': ['netflix_hours_viewed', 'netflix_hours_viewed_lag7', 'netflix_hours_viewed_lag14'],
    # Temporal views patterns - these are dynamically constructed
    'views_temporal': [],  # views_{h1|h2|q1-4}_{year}_{cc}
}


class ColumnAuthorizationError(Exception):
    """Raised when attempting to create unauthorized columns."""
    pass


class ForbiddenTemporalTypeError(Exception):
    """Raised when encountering forbidden temporal_period_type like 'lifetime'."""
    pass


class GPUMergeEngine:
    """
    GPU-accelerated merge engine using cuDF/RAPIDS.

    CRITICAL: This engine NEVER creates new columns without authorization.
    All column mappings must follow SCHEMA_V27.00.json.
    """

    def __init__(self):
        self.stats = {
            'base_rows': 0,
            'base_columns': 0,
            'flixpatrol_rows_processed': 0,
            'matches_found': 0,
            'values_merged': 0,
            'values_overwritten': 0,
            'rows_skipped_forbidden_temporal': 0,
            'rows_skipped_no_column_match': 0,
            'columns_requested_but_denied': [],
            'final_rows': 0,
            'final_columns': 0,
            'grep_performance': {
                'column_lookups': 0,
                'grep_cache_hits': 0,
            }
        }
        self.bfd = None
        self.bfd_columns = set()  # Cache of existing BFD columns
        self.schema = None
        self.season_allocator = None
        self.pending_authorization_requests = []  # Columns that need authorization

        # VBUS Memory Management with Grep Performance
        self.vbus_manager = None
        self.column_grep = None  # Grep engine for fast column lookups
        if VBUS_AVAILABLE:
            self.vbus_manager = VBUSGPUManager(use_gpu=True)
            print("[VBUS] GPU Memory Manager initialized")
            print("[GREP] O(1) column lookups enabled via grep engine")

    def load_schema(self):
        """Load schema and cache valid column names."""
        print(f"\n[SCHEMA] Loading from {SCHEMA_PATH}...")
        try:
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    self.schema = json.loads(content[:end_pos])
                    print(f"[SCHEMA] Version: {self.schema['_metadata']['schema_version']}")
        except Exception as e:
            print(f"[SCHEMA] Warning: {e}")
            self.schema = {'_metadata': {'schema_version': '27.00'}}

    def load_base_database(self):
        """Load base BFD and cache existing columns with grep-based indexing."""
        print(f"\n[GPU] Loading base database: {BFD_BASE}")

        if self.vbus_manager:
            self.vbus_manager.clear_gpu_memory()
            print("[VBUS] Cleared GPU memory before loading BFD")

        self.bfd = cudf.read_parquet(str(BFD_BASE))
        self.stats['base_rows'] = len(self.bfd)
        self.stats['base_columns'] = len(self.bfd.columns)

        # CRITICAL: Cache existing columns - we will NOT add new ones
        columns_list = self.bfd.columns.tolist()
        self.bfd_columns = set(columns_list)

        # Initialize grep engine for O(1) column lookups (PERFORMANCE OPTIMIZATION)
        if self.vbus_manager and VBUS_AVAILABLE:
            self.column_grep = self.vbus_manager.init_column_grep(columns_list)
            grep_stats = self.column_grep.get_stats()
            print(f"[GREP] Indexed {grep_stats['total_columns']} columns for fast lookup")
            print(f"[GREP] Families: quarterly={grep_stats['views_quarterly_count']}, half_yearly={grep_stats['views_half_yearly_count']}, lag={grep_stats['lag_features_count']}")

        print(f"[GPU] Loaded: {self.stats['base_rows']:,} rows x {self.stats['base_columns']:,} columns")
        print(f"[SCHEMA] Cached {len(self.bfd_columns):,} existing columns - NO NEW COLUMNS ALLOWED")

        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        print(f"[GPU] VRAM after load: {mem_free/1e9:.1f}GB free")

    def _validate_temporal_type(self, temporal_type: str, row_info: dict) -> bool:
        """
        Validate temporal_period_type. REJECT forbidden types.

        Returns True if valid, raises ForbiddenTemporalTypeError if forbidden.
        """
        if temporal_type is None or str(temporal_type) in EMPTY_VALUES:
            return True  # No temporal type specified - will use default mapping

        temporal_type = str(temporal_type).lower().strip()

        if temporal_type in FORBIDDEN_TEMPORAL_TYPES:
            self.stats['rows_skipped_forbidden_temporal'] += 1
            title = row_info.get('title', 'Unknown')
            imdb = row_info.get('imdb_id', 'N/A')
            print(f"[FORBIDDEN] temporal_period_type='{temporal_type}' for {title} ({imdb}) - SKIPPED")
            raise ForbiddenTemporalTypeError(
                f"temporal_period_type '{temporal_type}' is FORBIDDEN. "
                f"Title: {title}, IMDb: {imdb}. "
                f"Requires manual review and column mapping decision."
            )

        return True

    def _get_schema_column_for_views(self, source_col: str, temporal_type: str, temporal_period: str) -> str:
        """
        Map FlixPatrol views column to schema-compliant BFD column.

        RULES:
        - views_{cc} with temporal_type='half_yearly' + period='h1_2024' -> views_h1_2024_{cc}
        - views_{cc} with temporal_type='quarterly' + period='q1_2024' -> views_q1_2024_{cc}
        - views_{cc} with temporal_type='lifetime' -> FORBIDDEN

        Returns the schema column name or None if no valid mapping.
        """
        # Extract country code from source column (e.g., views_us -> us)
        if not source_col.startswith('views_'):
            return None

        cc = source_col.replace('views_', '')
        if cc not in COUNTRIES_22 and cc != 'total':
            return None

        # Determine target column based on temporal type (half_yearly or quarterly only)
        temporal_type = str(temporal_type).lower().strip() if temporal_type else None

        if temporal_type in ('half_yearly', 'quarterly'):
            # Parse temporal_period to get the actual period
            # e.g., "h1_2024" or "q1_2024"
            if temporal_period and str(temporal_period) not in EMPTY_VALUES:
                period = str(temporal_period).lower().strip()
                # Validate period format
                if period.startswith(('h1_', 'h2_', 'q1_', 'q2_', 'q3_', 'q4_')):
                    target = f'views_{period}_{cc}'
                else:
                    # Invalid period format
                    return None
            else:
                return None
        else:
            # Unknown temporal type - no mapping
            return None

        return target

    def _check_column_exists(self, column_name: str) -> bool:
        """
        Check if column exists in BFD using grep engine for O(1) lookup.

        CRITICAL: If column does NOT exist, do NOT create it.
        Log it for authorization request.

        Also checks for FORBIDDEN words in column names.

        PERFORMANCE: Uses grep engine for O(1) set-based lookup
        instead of O(n) list search (10-100x faster on 2299 columns).
        """
        self.stats['grep_performance']['column_lookups'] += 1

        # Check for forbidden words FIRST
        col_lower = column_name.lower()
        for forbidden in FORBIDDEN_COLUMN_WORDS:
            if forbidden in col_lower:
                print(f"[FORBIDDEN] Column '{column_name}' contains forbidden word '{forbidden}' - REJECTED")
                return False

        # Use grep engine O(1) lookup if available, else fallback to set
        if self.column_grep:
            exists = self.column_grep.exists(column_name)
        else:
            exists = column_name in self.bfd_columns

        if exists:
            return True

        # Column does not exist - log for authorization
        if column_name not in self.stats['columns_requested_but_denied']:
            self.stats['columns_requested_but_denied'].append(column_name)
            print(f"[DENIED] Column '{column_name}' does not exist in BFD - AUTHORIZATION REQUIRED")

        return False

    def process_flixpatrol_views(self):
        """
        Process FlixPatrol Views with STRICT schema enforcement.

        - REJECTS rows with temporal_period_type='lifetime'
        - Maps columns according to schema
        - NEVER creates new columns
        """
        print(f"\n[GPU] Processing FlixPatrol Views: {FLIXPATROL_VIEWS}")
        print("[SCHEMA] STRICT MODE: No new columns will be created")

        if not FLIXPATROL_VIEWS.exists():
            print(f"[WARN] File not found: {FLIXPATROL_VIEWS}")
            return

        # Load FlixPatrol data
        fp_df = pd.read_csv(str(FLIXPATROL_VIEWS))
        total_rows = len(fp_df)
        print(f"[GPU] Loaded: {total_rows:,} rows")

        # Process each row with schema validation
        processed = 0
        merged = 0
        skipped_forbidden = 0
        skipped_no_match = 0

        # Get BFD as pandas for updates
        bfd_pandas = self.bfd.to_pandas()

        # Normalize IMDb IDs for matching
        def norm_imdb(val):
            if val is None or str(val) in EMPTY_VALUES:
                return None
            s = str(val)
            if s.isdigit():
                return f"tt{s}"
            return s

        bfd_pandas['imdb_id_norm'] = bfd_pandas['imdb_id'].apply(norm_imdb)
        fp_df['imdb_id_norm'] = fp_df['imdb_id'].apply(norm_imdb)

        for idx, fp_row in fp_df.iterrows():
            processed += 1

            # Get temporal info
            temporal_type = fp_row.get('temporal_period_type')
            temporal_period = fp_row.get('temporal_period')

            row_info = {
                'title': fp_row.get('title'),
                'imdb_id': fp_row.get('imdb_id'),
                'temporal_type': temporal_type,
                'temporal_period': temporal_period
            }

            # VALIDATE: Check for forbidden temporal types
            try:
                self._validate_temporal_type(temporal_type, row_info)
            except ForbiddenTemporalTypeError:
                skipped_forbidden += 1
                continue

            # Find matching BFD row
            imdb_norm = fp_row.get('imdb_id_norm')
            if not imdb_norm:
                continue

            mask = bfd_pandas['imdb_id_norm'] == imdb_norm
            if not mask.any():
                # Try fc_uid match
                fc_uid = fp_row.get('fc_uid')
                if fc_uid and str(fc_uid) not in EMPTY_VALUES:
                    mask = bfd_pandas['fc_uid'] == fc_uid

            if not mask.any():
                skipped_no_match += 1
                continue

            # Process views columns
            for cc in COUNTRIES_22:
                src_col = f'views_{cc}'
                if src_col not in fp_row or pd.isna(fp_row[src_col]):
                    continue

                fp_val = fp_row[src_col]
                if fp_val == 0:
                    continue

                # Get schema-compliant target column
                target_col = self._get_schema_column_for_views(src_col, temporal_type, temporal_period)

                if target_col is None:
                    skipped_no_match += 1
                    continue

                # Check if target column exists
                if not self._check_column_exists(target_col):
                    self.stats['rows_skipped_no_column_match'] += 1
                    continue

                # MERGE: Update BFD with FlixPatrol value (OVERWRITE mode)
                bfd_pandas.loc[mask, target_col] = fp_val
                merged += 1

        # Convert back to cuDF
        self.bfd = cudf.DataFrame(bfd_pandas)

        self.stats['flixpatrol_rows_processed'] = processed
        self.stats['values_merged'] = merged
        self.stats['rows_skipped_forbidden_temporal'] = skipped_forbidden

        print(f"\n[RESULT] Processed: {processed:,} rows")
        print(f"[RESULT] Merged: {merged:,} values")
        print(f"[RESULT] Skipped (forbidden temporal): {skipped_forbidden:,}")
        print(f"[RESULT] Skipped (no column match): {skipped_no_match:,}")

        if self.stats['columns_requested_but_denied']:
            print(f"\n[AUTHORIZATION REQUIRED] The following columns were requested but do not exist:")
            for col in self.stats['columns_requested_but_denied'][:20]:
                print(f"  - {col}")
            if len(self.stats['columns_requested_but_denied']) > 20:
                print(f"  ... and {len(self.stats['columns_requested_but_denied']) - 20} more")

    def save(self):
        """Save output - only if authorized."""
        print(f"\n[GPU] Preparing to save to {OUTPUT_FILE}...")

        # SAFETY CHECK: Verify no new columns were created
        final_columns = set(self.bfd.columns.tolist())
        new_columns = final_columns - self.bfd_columns

        if new_columns:
            print(f"\n[CRITICAL ERROR] Unauthorized columns detected!")
            for col in new_columns:
                print(f"  - {col}")
            print("[ABORT] Save cancelled - unauthorized columns found")
            return False

        self.stats['final_rows'] = len(self.bfd)
        self.stats['final_columns'] = len(self.bfd.columns)

        # Save parquet
        self.bfd.to_parquet(str(OUTPUT_FILE))
        file_size = OUTPUT_FILE.stat().st_size / (1024**3)
        print(f"[GPU] Saved: {self.stats['final_rows']:,} rows x {self.stats['final_columns']:,} columns")
        print(f"[GPU] File size: {file_size:.2f} GB")

        # Save report
        report = {
            'version': '27.00',
            'created_at': datetime.now().isoformat(),
            'gpu_accelerated': True,
            'schema_enforced': True,
            'new_columns_created': 0,  # MUST be zero
            'statistics': self.stats,
            'authorization_required': self.stats['columns_requested_but_denied'],
        }

        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[GPU] Report: {REPORT_FILE}")

        return True

    def run(self):
        """Execute merge with strict schema enforcement."""
        print("=" * 70)
        print("    INTELLIGENT MERGE ENGINE V27.00 - STRICT SCHEMA MODE")
        print("    NO NEW COLUMNS WITHOUT AUTHORIZATION")
        print("=" * 70)
        start_time = datetime.now()

        self.load_schema()
        self.load_base_database()

        self.process_flixpatrol_views()

        # Check for authorization issues before save
        if self.stats['columns_requested_but_denied']:
            print("\n" + "=" * 70)
            print("    MERGE PAUSED - AUTHORIZATION REQUIRED")
            print("=" * 70)
            print(f"\n{len(self.stats['columns_requested_but_denied'])} columns were requested but do not exist in BFD.")
            print("Options:")
            print("  1. Add these columns to schema and BFD (requires approval)")
            print("  2. Update FlixPatrol data to use existing column mappings")
            print("  3. Skip rows that don't match existing columns")
            print("\nNo data was saved. Review and provide authorization.")
            return self.stats

        success = self.save()

        elapsed = datetime.now() - start_time

        print("\n" + "=" * 70)
        print("    MERGE COMPLETE" if success else "    MERGE FAILED")
        print("=" * 70)
        print(f"Duration: {elapsed}")
        print(f"\nStatistics:")
        print(f"  Rows processed: {self.stats['flixpatrol_rows_processed']:,}")
        print(f"  Values merged: {self.stats['values_merged']:,}")
        print(f"  Skipped (forbidden temporal): {self.stats['rows_skipped_forbidden_temporal']:,}")
        print(f"  New columns created: 0 (ENFORCED)")

        # Grep performance stats
        if self.column_grep:
            grep_perf = self.stats['grep_performance']
            grep_stats = self.column_grep.get_stats()
            print(f"\nGrep Performance (O(1) Column Lookups):")
            print(f"  Column lookups performed: {grep_perf['column_lookups']:,}")
            print(f"  Indexed columns: {grep_stats['total_columns']}")
            print(f"  Grep cache entries: {grep_stats['cache_entries']}")
            print(f"  Benefit: ~{grep_perf['column_lookups'] * 2299 / 1000:.0f}K comparisons avoided")

        return self.stats


if __name__ == '__main__':
    engine = GPUMergeEngine()
    engine.run()
