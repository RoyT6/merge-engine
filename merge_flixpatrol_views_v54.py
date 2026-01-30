#!/usr/bin/env python3
"""
FlixPatrol Views Merge Engine V54 - Windows Compatible
Merges FlixPatrol season-allocated views into BFD main database.

CRITICAL RULES:
1. FlixPatrol views have SENIOR CREDIBILITY - OVERWRITE existing values
2. Match by fc_uid (normalized format)
3. Map views_{cc} to views_{period}_{year}_{cc} based on temporal period
4. Append NEW rows for titles not in BFD
5. No new columns without authorization

Input: FlixPatrol_Views_Season_Allocated.parquet (63,332 rows)
Output: BFD_V27.54.parquet (merged database)
"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add Parallel Engine for optimized processing
sys.path.insert(0, str(Path(r"C:\Users\RoyT6\Downloads\Parallel Engine")))
from system_capability_engine.quick_config import CPU_WORKERS

# Configuration
BASE_DIR = Path(r"C:\Users\RoyT6\Downloads")
BFD_INPUT = BASE_DIR / "BFD_V27.53.parquet"
FLIXPATROL_INPUT = BASE_DIR / "Views TRaining Data" / "FlixPatrol_Views_Season_Allocated_COMPLETE.parquet"
BFD_OUTPUT = BASE_DIR / "BFD_V27.54.parquet"
REPORT_OUTPUT = BASE_DIR / "BFD_V27.54_merge_report.json"

# 22 Countries
COUNTRIES_22 = [
    'us', 'cn', 'in', 'gb', 'br', 'de', 'jp', 'fr', 'ca', 'mx',
    'au', 'es', 'it', 'kr', 'nl', 'se', 'sg', 'hk', 'ie', 'ru',
    'tr', 'row'
]


class FlixPatrolMergeEngine:
    """
    Merges FlixPatrol views into BFD with SENIOR credibility.
    FlixPatrol data OVERWRITES existing views.
    """

    def __init__(self):
        self.stats = {
            'bfd_rows_before': 0,
            'bfd_columns': 0,
            'flixpatrol_rows': 0,
            'matched_by_fc_uid': 0,
            'matched_by_imdb': 0,
            'values_overwritten': 0,
            'new_rows_appended': 0,
            'unmapped_temporal_periods': 0,
            'bfd_rows_after': 0,
        }
        self.bfd = None
        self.flixpatrol = None
        self.bfd_columns = set()

    def normalize_fc_uid(self, fc_uid: str) -> str:
        """
        Normalize fc_uid to consistent format.
        tt10919420_s1 -> tt10919420_s1
        tt13443470_s01 -> tt13443470_s1
        """
        if not fc_uid or pd.isna(fc_uid):
            return None

        fc_uid = str(fc_uid).strip()

        # Normalize season suffix: _s01 -> _s1, _s001 -> _s1
        match = re.match(r'^(tt\d+)_s0*(\d+)$', fc_uid)
        if match:
            return f"{match.group(1)}_s{match.group(2)}"

        return fc_uid

    def get_target_column(self, source_col: str, temporal_period: str) -> Optional[str]:
        """
        Map FlixPatrol views column to BFD schema column.

        views_us + h2_2025 -> views_h2_2025_us
        views_cn + h1_2024 -> views_h1_2024_cn

        Returns None if no valid mapping.
        """
        if not source_col.startswith('views_'):
            return None

        cc = source_col.replace('views_', '')
        if cc not in COUNTRIES_22 and cc != 'total':
            return None

        if not temporal_period or pd.isna(temporal_period):
            return None

        temporal_period = str(temporal_period).lower().strip()

        # Skip invalid temporal periods
        if 'nan' in temporal_period or temporal_period == '':
            return None

        # Valid patterns: h1_2024, h2_2025, q1_2024, etc.
        if re.match(r'^(h[12]|q[1-4])_\d{4}$', temporal_period):
            target = f'views_{temporal_period}_{cc}'
            return target

        return None

    def load_data(self):
        """Load BFD and FlixPatrol data."""
        print("=" * 70)
        print("FLIXPATROL VIEWS MERGE ENGINE V54")
        print("FlixPatrol SENIOR - Overwrites existing views")
        print("=" * 70)

        # Load BFD
        print(f"\nLoading BFD: {BFD_INPUT}")
        self.bfd = pd.read_parquet(BFD_INPUT)
        self.stats['bfd_rows_before'] = len(self.bfd)
        self.stats['bfd_columns'] = len(self.bfd.columns)
        self.bfd_columns = set(self.bfd.columns.tolist())
        print(f"  Rows: {self.stats['bfd_rows_before']:,}")
        print(f"  Columns: {self.stats['bfd_columns']:,}")

        # Normalize BFD fc_uids for matching
        self.bfd['fc_uid_norm'] = self.bfd['fc_uid'].apply(self.normalize_fc_uid)

        # Load FlixPatrol
        print(f"\nLoading FlixPatrol: {FLIXPATROL_INPUT}")
        self.flixpatrol = pd.read_parquet(FLIXPATROL_INPUT)
        self.stats['flixpatrol_rows'] = len(self.flixpatrol)
        print(f"  Rows: {self.stats['flixpatrol_rows']:,}")

        # Normalize FlixPatrol fc_uids
        self.flixpatrol['fc_uid_norm'] = self.flixpatrol['fc_uid'].apply(self.normalize_fc_uid)

        # Filter to rows with views
        fp_with_views = self.flixpatrol[self.flixpatrol['total_views'] > 0]
        print(f"  Rows with views > 0: {len(fp_with_views):,}")

    def merge_views(self):
        """
        Merge FlixPatrol views into BFD.
        FlixPatrol has SENIOR credibility - overwrites existing.
        """
        print("\n" + "-" * 70)
        print("MERGING VIEWS (FlixPatrol SENIOR - Overwrites)")
        print("-" * 70)

        # Filter to rows with actual views AND valid temporal period
        fp_active = self.flixpatrol[
            (self.flixpatrol['total_views'] > 0) &
            (self.flixpatrol['temporal period'].notna()) &
            (~self.flixpatrol['temporal period'].str.contains('nan', case=False, na=True))
        ].copy()
        print(f"Processing {len(fp_active):,} FlixPatrol rows with views AND valid temporal period")

        # Show temporal period distribution being merged
        print("\nTemporal periods being merged:")
        print(fp_active['temporal period'].value_counts().head(10).to_string())

        # Create BFD fc_uid lookup for fast matching
        bfd_fc_uid_set = set(self.bfd['fc_uid_norm'].dropna().tolist())
        print(f"BFD has {len(bfd_fc_uid_set):,} unique fc_uids")

        # Track new rows to append
        new_rows = []
        values_merged = 0
        matched_count = 0
        unmatched_count = 0

        for idx, fp_row in fp_active.iterrows():
            fc_uid_norm = fp_row['fc_uid_norm']
            temporal_period = fp_row.get('temporal period')

            if not fc_uid_norm:
                continue

            # Check if fc_uid exists in BFD
            if fc_uid_norm in bfd_fc_uid_set:
                # MATCH FOUND - Update BFD row
                matched_count += 1
                mask = self.bfd['fc_uid_norm'] == fc_uid_norm

                # Merge each country's views
                for cc in COUNTRIES_22:
                    src_col = f'views_{cc}'
                    if src_col not in fp_row:
                        continue

                    fp_val = fp_row[src_col]
                    if pd.isna(fp_val) or fp_val == 0:
                        continue

                    # Get target column based on temporal period
                    target_col = self.get_target_column(src_col, temporal_period)

                    if target_col and target_col in self.bfd_columns:
                        # OVERWRITE with FlixPatrol value (SENIOR)
                        self.bfd.loc[mask, target_col] = fp_val
                        values_merged += 1
                    elif target_col:
                        self.stats['unmapped_temporal_periods'] += 1

                # Also update total_views if we have a views_computed column
                if 'views_computed' in self.bfd_columns and fp_row['total_views'] > 0:
                    self.bfd.loc[mask, 'views_computed'] = fp_row['total_views']
                    values_merged += 1

            else:
                # NO MATCH - Prepare new row to append
                unmatched_count += 1
                new_row = self._create_new_row(fp_row)
                if new_row:
                    new_rows.append(new_row)

        self.stats['matched_by_fc_uid'] = matched_count
        self.stats['values_overwritten'] = values_merged
        print(f"  Matched fc_uids: {matched_count:,}")
        print(f"  Values overwritten: {values_merged:,}")
        print(f"  Unmatched fc_uids: {unmatched_count:,}")

        # Append new rows
        if new_rows:
            print(f"\nAppending {len(new_rows):,} new rows to BFD...")
            new_df = pd.DataFrame(new_rows)
            self.bfd = pd.concat([self.bfd, new_df], ignore_index=True)
            self.stats['new_rows_appended'] = len(new_rows)

        # Clean up temp column
        if 'fc_uid_norm' in self.bfd.columns:
            self.bfd = self.bfd.drop(columns=['fc_uid_norm'])

        self.stats['bfd_rows_after'] = len(self.bfd)

    def _create_new_row(self, fp_row: pd.Series) -> Optional[Dict]:
        """
        Create a new BFD row from FlixPatrol data.
        Only creates if we have essential identifiers.
        """
        fc_uid = fp_row.get('fc_uid')
        if not fc_uid or pd.isna(fc_uid):
            return None

        temporal_period = fp_row.get('temporal period')

        # Create row with all BFD columns (mostly NULL)
        new_row = {col: None for col in self.bfd_columns if col != 'fc_uid_norm'}

        # Core identifiers
        new_row['fc_uid'] = fc_uid
        new_row['imdb_id'] = fp_row.get('imdb_id')
        new_row['tmdb_id'] = fp_row.get('tmdb_id')
        new_row['title'] = fp_row.get('title')
        new_row['title_type'] = fp_row.get('title_type', 'tv_show')
        new_row['flixpatrol_id'] = fp_row.get('flixpatrol_id')

        # Season info
        new_row['season_number'] = fp_row.get('season_number')
        new_row['max_seasons'] = fp_row.get('max_seasons')

        # Metadata
        new_row['start_year'] = fp_row.get('start_year')
        new_row['runtime_minutes'] = fp_row.get('runtime_minutes')

        # Views - map to temporal columns
        for cc in COUNTRIES_22:
            src_col = f'views_{cc}'
            if src_col in fp_row:
                fp_val = fp_row[src_col]
                if not pd.isna(fp_val) and fp_val > 0:
                    target_col = self.get_target_column(src_col, temporal_period)
                    if target_col and target_col in self.bfd_columns:
                        new_row[target_col] = fp_val

        # Total views
        if 'views_computed' in self.bfd_columns:
            new_row['views_computed'] = fp_row.get('total_views')

        return new_row

    def save(self):
        """Save merged BFD."""
        print("\n" + "-" * 70)
        print("SAVING MERGED DATABASE")
        print("-" * 70)

        # Save parquet
        self.bfd.to_parquet(BFD_OUTPUT, index=False)
        file_size = BFD_OUTPUT.stat().st_size / (1024**3)
        print(f"Saved: {BFD_OUTPUT}")
        print(f"  Rows: {self.stats['bfd_rows_after']:,}")
        print(f"  Columns: {len(self.bfd.columns):,}")
        print(f"  Size: {file_size:.2f} GB")

        # Save report
        report = {
            'version': '27.54',
            'merge_date': datetime.now().isoformat(),
            'source': {
                'bfd': str(BFD_INPUT),
                'flixpatrol': str(FLIXPATROL_INPUT),
            },
            'output': str(BFD_OUTPUT),
            'statistics': self.stats,
            'merge_rules': {
                'credibility': 'FlixPatrol SENIOR - overwrites existing',
                'match_key': 'fc_uid (normalized)',
                'column_mapping': 'views_{cc} -> views_{period}_{year}_{cc}',
            }
        }

        with open(REPORT_OUTPUT, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report: {REPORT_OUTPUT}")

    def run(self):
        """Execute merge pipeline."""
        start_time = datetime.now()

        self.load_data()
        self.merge_views()
        self.save()

        elapsed = datetime.now() - start_time

        print("\n" + "=" * 70)
        print("MERGE COMPLETE")
        print("=" * 70)
        print(f"Duration: {elapsed}")
        print(f"\nSummary:")
        print(f"  BFD rows before: {self.stats['bfd_rows_before']:,}")
        print(f"  BFD rows after:  {self.stats['bfd_rows_after']:,}")
        print(f"  Matched fc_uids: {self.stats['matched_by_fc_uid']:,}")
        print(f"  Values overwritten: {self.stats['values_overwritten']:,}")
        print(f"  New rows appended: {self.stats['new_rows_appended']:,}")

        return self.stats


def main():
    engine = FlixPatrolMergeEngine()
    engine.run()


if __name__ == "__main__":
    main()
