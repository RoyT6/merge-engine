"""
FreshIn Merge Engine for BFD V28.00
===================================
Merges fresh data from FreshIn folder into master BFD database.

Schema V28.00 Compliant:
- NO NEW COLUMNS without authorization
- Updates existing records with fresh data
- Appends new titles not in BFD
- Follows INGESTION_MANIFEST rules

Data Sources Processed:
- TMDB trending/popular (updates: tmdb_popularity, tmdb_score, tmdb_vote_count)
- FlixPatrol rankings (updates: flixpatrol_points, flixpatrol_rank)
- Cranberry Fresh parquets (consolidated fresh scrapes)

Created: 2026-01-24
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Paths
BASE_DIR = Path("C:/Users/RoyT6/Downloads")
FRESHIN_DIR = BASE_DIR / "FreshIn"
SCHEMA_PATH = BASE_DIR / "Schema" / "SCHEMA_V28.00.json"
BFD_INPUT = BASE_DIR / "BFD_V26.02.parquet"
BFD_OUTPUT = BASE_DIR / "BFD_V26.03.parquet"
REPORT_PATH = BASE_DIR / "FreshIn_Merge_Report_V26.03.json"

# Empty value handling (from schema rules)
EMPTY_VALUES = {'', 'None', 'none', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', '<NA>'}


def is_empty(val) -> bool:
    """Check if value is effectively empty/null."""
    if val is None or pd.isna(val):
        return True
    if isinstance(val, str):
        return val.strip() in EMPTY_VALUES
    return False


class FreshInMergeEngine:
    """
    Merges FreshIn data into BFD master database.

    Rules:
    - NEVER creates new columns
    - Updates existing records (NULL-fill or overwrite based on source credibility)
    - Appends new titles with minimal required columns
    """

    def __init__(self):
        self.stats = {
            'bfd_rows_start': 0,
            'bfd_cols': 0,
            'tmdb_updates': 0,
            'tmdb_new_titles': 0,
            'flixpatrol_updates': 0,
            'cranberry_updates': 0,
            'total_values_updated': 0,
            'total_rows_appended': 0,
            'bfd_rows_end': 0,
            'processing_errors': [],
        }
        self.bfd = None
        self.bfd_columns = set()

    def load_bfd(self):
        """Load base BFD database."""
        print(f"\n[LOAD] Loading BFD from {BFD_INPUT}")
        self.bfd = pd.read_parquet(BFD_INPUT)
        self.stats['bfd_rows_start'] = len(self.bfd)
        self.stats['bfd_cols'] = len(self.bfd.columns)
        self.bfd_columns = set(self.bfd.columns)

        # Build TMDB ID index for fast lookups
        self.tmdb_index = {}
        for idx, row in self.bfd.iterrows():
            tmdb_id = row.get('tmdb_id')
            if not is_empty(tmdb_id):
                try:
                    self.tmdb_index[int(tmdb_id)] = idx
                except (ValueError, TypeError):
                    pass

        print(f"[LOAD] BFD: {self.stats['bfd_rows_start']:,} rows x {self.stats['bfd_cols']:,} columns")
        print(f"[LOAD] TMDB ID index: {len(self.tmdb_index):,} records")

    def load_fresh_tmdb(self) -> List[dict]:
        """Load all fresh TMDB data from FreshIn."""
        print("\n[TMDB] Loading fresh TMDB data...")

        all_items = []
        tmdb_files = sorted(FRESHIN_DIR.glob("fresh_data_*.json"), reverse=True)

        # Process most recent first, collect unique TMDB IDs
        seen_ids = set()
        for json_file in tmdb_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                results = data.get('results', [])
                for result in results:
                    if result.get('source') == 'tmdb' and result.get('data'):
                        items = result['data'].get('results', [])
                        for item in items:
                            tmdb_id = item.get('id')
                            if tmdb_id and tmdb_id not in seen_ids:
                                seen_ids.add(tmdb_id)
                                all_items.append({
                                    'tmdb_id': int(tmdb_id),
                                    'title': item.get('title') or item.get('name'),
                                    'original_title': item.get('original_title') or item.get('original_name'),
                                    'overview': item.get('overview'),
                                    'popularity': item.get('popularity'),
                                    'vote_average': item.get('vote_average'),
                                    'vote_count': item.get('vote_count'),
                                    'release_date': item.get('release_date') or item.get('first_air_date'),
                                    'media_type': item.get('media_type'),
                                    'original_language': item.get('original_language'),
                                    'genre_ids': item.get('genre_ids', []),
                                    'adult': item.get('adult', False),
                                    'poster_path': item.get('poster_path'),
                                    'backdrop_path': item.get('backdrop_path'),
                                    'source_file': json_file.name,
                                })
            except Exception as e:
                self.stats['processing_errors'].append(f"TMDB {json_file.name}: {str(e)}")

        print(f"[TMDB] Loaded {len(all_items):,} unique TMDB records from {len(tmdb_files)} files")
        return all_items

    def load_fresh_flixpatrol(self) -> List[dict]:
        """Load fresh FlixPatrol data."""
        print("\n[FLIXPATROL] Loading fresh FlixPatrol data...")

        all_items = []
        fp_files = sorted(FRESHIN_DIR.glob("flixpatrol_*.json"), reverse=True)

        for json_file in fp_files[:3]:  # Process 3 most recent
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for entry in data:
                        if entry.get('source') == 'flixpatrol':
                            fp_data = entry.get('data', {}).get('data', [])
                            for item in fp_data:
                                item_data = item.get('data', {})
                                if item_data.get('id'):
                                    all_items.append({
                                        'flixpatrol_id': item_data.get('id'),
                                        'title': item_data.get('title'),
                                        'link': item_data.get('link'),
                                        'premiere': item_data.get('premiere'),
                                        'source_file': json_file.name,
                                    })
            except Exception as e:
                self.stats['processing_errors'].append(f"FlixPatrol {json_file.name}: {str(e)}")

        print(f"[FLIXPATROL] Loaded {len(all_items):,} FlixPatrol records")
        return all_items

    def load_cranberry_fresh(self) -> pd.DataFrame:
        """Load and consolidate Cranberry Fresh parquet files."""
        print("\n[CRANBERRY] Loading Cranberry Fresh parquets...")

        dfs = []
        parquet_files = sorted(FRESHIN_DIR.glob("cranberry_fresh_*.parquet"), reverse=True)

        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                df['source_file'] = pf.name
                dfs.append(df)
            except Exception as e:
                self.stats['processing_errors'].append(f"Cranberry {pf.name}: {str(e)}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            # Deduplicate by id, keeping most recent
            combined = combined.drop_duplicates(subset=['id'], keep='first')
            print(f"[CRANBERRY] Loaded {len(combined):,} unique records from {len(parquet_files)} files")
            return combined
        else:
            return pd.DataFrame()

    def process_tmdb_updates(self, tmdb_items: List[dict]):
        """
        Update BFD records with fresh TMDB data.

        Updates (if fresher/better data):
        - tmdb_popularity
        - tmdb_score (from vote_average)
        - tmdb_vote_count
        """
        print("\n[MERGE] Processing TMDB updates...")

        updates = 0
        values_updated = 0
        new_titles = []

        for item in tmdb_items:
            tmdb_id = item['tmdb_id']

            if tmdb_id in self.tmdb_index:
                # UPDATE existing record
                idx = self.tmdb_index[tmdb_id]

                # Update tmdb_popularity (always take fresh)
                if item.get('popularity') is not None:
                    old_val = self.bfd.at[idx, 'tmdb_popularity']
                    new_val = float(item['popularity'])
                    if is_empty(old_val) or new_val != old_val:
                        self.bfd.at[idx, 'tmdb_popularity'] = new_val
                        values_updated += 1

                # Update tmdb_score (from vote_average)
                if item.get('vote_average') is not None:
                    if 'tmdb_score' in self.bfd_columns:
                        old_val = self.bfd.at[idx, 'tmdb_score']
                        new_val = float(item['vote_average'])
                        if is_empty(old_val) or new_val != old_val:
                            self.bfd.at[idx, 'tmdb_score'] = new_val
                            values_updated += 1

                # Update overview if empty
                if item.get('overview') and 'overview' in self.bfd_columns:
                    old_val = self.bfd.at[idx, 'overview']
                    if is_empty(old_val):
                        self.bfd.at[idx, 'overview'] = item['overview']
                        values_updated += 1

                updates += 1
            else:
                # NEW title - prepare for append
                new_titles.append(item)

        self.stats['tmdb_updates'] = updates
        self.stats['total_values_updated'] += values_updated

        print(f"[MERGE] TMDB updates: {updates:,} records, {values_updated:,} values")
        print(f"[MERGE] TMDB new titles: {len(new_titles):,}")

        return new_titles

    def append_new_titles(self, new_titles: List[dict]):
        """
        Append new titles to BFD.

        Creates minimal rows with required columns.
        Does NOT create new columns.
        Type-safe: matches dtypes from existing BFD.
        """
        if not new_titles:
            print("[APPEND] No new titles to append")
            return

        print(f"\n[APPEND] Appending {len(new_titles):,} new titles...")

        # Get dtypes from existing BFD for type consistency
        bfd_dtypes = self.bfd.dtypes.to_dict()

        new_rows = []
        for item in new_titles:
            # Create row with only existing columns, using None
            row = {col: None for col in self.bfd_columns}

            # Map TMDB data to BFD columns
            row['tmdb_id'] = float(item['tmdb_id'])
            row['title'] = str(item.get('title', '')) if item.get('title') else None
            row['original_title'] = str(item.get('original_title', '')) if item.get('original_title') else None
            row['overview'] = str(item.get('overview', '')) if item.get('overview') else None
            row['tmdb_popularity'] = float(item['popularity']) if item.get('popularity') is not None else None
            row['tmdb_score'] = float(item['vote_average']) if item.get('vote_average') is not None else None
            row['original_language'] = str(item.get('original_language', '')) if item.get('original_language') else None
            row['is_adult'] = str(item.get('adult', False))  # BFD stores as string "True"/"False"
            row['poster'] = f"https://image.tmdb.org/t/p/original{item['poster_path']}" if item.get('poster_path') else None
            row['backdrop'] = f"https://image.tmdb.org/t/p/original{item['backdrop_path']}" if item.get('backdrop_path') else None

            # Set title_type based on media_type
            media_type = item.get('media_type')
            if media_type == 'movie':
                row['title_type'] = 'movie'
            elif media_type == 'tv':
                row['title_type'] = 'tv_show'
                row['season_number'] = 1.0
                row['max_seasons'] = 1.0
            else:
                row['title_type'] = 'movie'  # Default

            # Parse release date to start_year
            release_date = item.get('release_date')
            if release_date:
                try:
                    year = int(release_date[:4])
                    row['start_year'] = float(year)
                    row['premiere_date'] = str(release_date)
                except:
                    pass

            # Generate fc_uid
            if row['title_type'] == 'tv_show':
                row['fc_uid'] = f"tmdb_{item['tmdb_id']}_s01"
            else:
                row['fc_uid'] = f"tmdb_{item['tmdb_id']}"

            # Ingestion metadata
            row['source_api'] = 'tmdb'
            row['ingestion_timestamp'] = datetime.now().isoformat()

            new_rows.append(row)

        # Create DataFrame with matching dtypes
        new_df = pd.DataFrame(new_rows, columns=list(self.bfd_columns))

        # Ensure column order matches
        new_df = new_df[self.bfd.columns]

        # Cast to matching dtypes where possible
        for col in new_df.columns:
            try:
                if col in bfd_dtypes:
                    dtype = bfd_dtypes[col]
                    if pd.api.types.is_float_dtype(dtype):
                        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    elif pd.api.types.is_integer_dtype(dtype):
                        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    elif pd.api.types.is_bool_dtype(dtype):
                        new_df[col] = new_df[col].astype('bool')
            except Exception:
                pass  # Keep original type if conversion fails

        # Append to BFD
        self.bfd = pd.concat([self.bfd, new_df], ignore_index=True)

        self.stats['total_rows_appended'] = len(new_rows)
        self.stats['tmdb_new_titles'] = len(new_rows)
        print(f"[APPEND] Added {len(new_rows):,} new rows to BFD")

    def save(self):
        """Save updated BFD."""
        print(f"\n[SAVE] Saving to {BFD_OUTPUT}...")

        self.stats['bfd_rows_end'] = len(self.bfd)

        # Verify no new columns created
        final_columns = set(self.bfd.columns)
        new_columns = final_columns - self.bfd_columns
        if new_columns:
            print(f"[ERROR] Unauthorized new columns detected: {new_columns}")
            print("[ABORT] Not saving - unauthorized columns")
            return False

        # Save parquet with explicit engine for better compatibility
        self.bfd.to_parquet(BFD_OUTPUT, index=False, engine='pyarrow', compression='snappy')
        file_size = BFD_OUTPUT.stat().st_size / (1024**3)
        print(f"[SAVE] Saved: {self.stats['bfd_rows_end']:,} rows x {len(self.bfd.columns):,} columns")
        print(f"[SAVE] File size: {file_size:.2f} GB")

        # Save report
        report = {
            'version': 'V26.03',
            'schema': 'V28.00',
            'created_at': datetime.now().isoformat(),
            'input_file': str(BFD_INPUT),
            'output_file': str(BFD_OUTPUT),
            'statistics': self.stats,
            'data_sources': {
                'tmdb': 'fresh_data_*.json',
                'flixpatrol': 'flixpatrol_*.json',
                'cranberry': 'cranberry_fresh_*.parquet',
            }
        }

        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[SAVE] Report: {REPORT_PATH}")

        return True

    def run(self):
        """Execute full merge pipeline."""
        print("=" * 70)
        print("    FRESHIN MERGE ENGINE V28.00")
        print("    Schema V28.00 Compliant - NO NEW COLUMNS")
        print("=" * 70)

        start_time = datetime.now()

        # Load BFD
        self.load_bfd()

        # Load fresh data
        tmdb_items = self.load_fresh_tmdb()
        # fp_items = self.load_fresh_flixpatrol()  # Disabled - no direct merge path
        # cranberry_df = self.load_cranberry_fresh()  # Disabled - needs ID mapping

        # Process updates
        new_titles = self.process_tmdb_updates(tmdb_items)

        # Append new titles
        self.append_new_titles(new_titles)

        # Save
        success = self.save()

        elapsed = datetime.now() - start_time

        print("\n" + "=" * 70)
        print("    MERGE COMPLETE" if success else "    MERGE FAILED")
        print("=" * 70)
        print(f"Duration: {elapsed}")
        print(f"\nStatistics:")
        print(f"  BFD rows: {self.stats['bfd_rows_start']:,} -> {self.stats['bfd_rows_end']:,}")
        print(f"  TMDB updates: {self.stats['tmdb_updates']:,}")
        print(f"  New titles appended: {self.stats['total_rows_appended']:,}")
        print(f"  Values updated: {self.stats['total_values_updated']:,}")

        if self.stats['processing_errors']:
            print(f"\nProcessing Errors ({len(self.stats['processing_errors'])}):")
            for err in self.stats['processing_errors'][:5]:
                print(f"  - {err}")

        return self.stats


if __name__ == '__main__':
    engine = FreshInMergeEngine()
    engine.run()
