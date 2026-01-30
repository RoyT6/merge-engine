#!/usr/bin/env python3
"""
List all BFD V26 columns organized by type.
Uses pyarrow for fast metadata-only read.
"""
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
import json

BFD_PATH = Path("/mnt/c/Users/RoyT6/Downloads/BFD_V26.00.parquet")
OUTPUT_PATH = Path("/mnt/c/Users/RoyT6/Downloads/Merge Engine/BFD_V26_columns_by_type.json")

print("Loading BFD V26.00 schema (metadata only)...")
parquet_file = pq.ParquetFile(str(BFD_PATH))
schema = parquet_file.schema_arrow

class MockDF:
    def __init__(self, columns):
        self.columns = columns

df = MockDF([field.name for field in schema])
num_rows = parquet_file.metadata.num_rows
print(f"Schema loaded: {num_rows:,} rows, {len(df.columns):,} columns")

# Categorize columns
categories = defaultdict(list)

for col in sorted(df.columns):
    col_lower = col.lower()

    # Views columns
    if col_lower.startswith('views_'):
        if any(f'_h{i}_' in col_lower for i in [1,2]):
            categories['views_half_yearly'].append(col)
        elif any(f'_q{i}_' in col_lower for i in [1,2,3,4]):
            categories['views_quarterly'].append(col)
        elif any(f'_m{str(i).zfill(2)}_' in col_lower for i in range(1,13)):
            categories['views_monthly'].append(col)
        else:
            categories['views_other'].append(col)

    # Hours columns
    elif col_lower.startswith('hours_'):
        categories['hours'].append(col)

    # FlixPatrol columns
    elif 'flixpatrol' in col_lower:
        categories['flixpatrol'].append(col)

    # Parrot columns
    elif col_lower.startswith('parrot_'):
        categories['parrot'].append(col)

    # Metacritic columns
    elif 'metacritic' in col_lower:
        categories['metacritic'].append(col)

    # IMDB columns
    elif 'imdb' in col_lower:
        categories['imdb'].append(col)

    # TMDB columns
    elif 'tmdb' in col_lower:
        categories['tmdb'].append(col)

    # RT columns
    elif col_lower.startswith('rt_') or 'rotten' in col_lower:
        categories['rotten_tomatoes'].append(col)

    # Nielsen columns
    elif 'nielsen' in col_lower:
        categories['nielsen'].append(col)

    # Streaming columns
    elif 'streaming' in col_lower or 'platform' in col_lower:
        categories['streaming'].append(col)

    # Netflix columns
    elif 'netflix' in col_lower:
        categories['netflix'].append(col)

    # Core identity columns
    elif col_lower in ['fc_uid', 'title', 'title_type', 'original_title', 'year', 'start_year', 'end_year']:
        categories['core_identity'].append(col)

    # Season columns
    elif 'season' in col_lower:
        categories['season'].append(col)

    # Genre columns
    elif 'genre' in col_lower:
        categories['genre'].append(col)

    # Country/region columns
    elif 'country' in col_lower or 'region' in col_lower:
        categories['country_region'].append(col)

    # Cast/crew columns
    elif any(x in col_lower for x in ['cast', 'director', 'actor', 'writer', 'producer', 'crew']):
        categories['cast_crew'].append(col)

    # Other
    else:
        categories['other'].append(col)

# Summary
print("\n" + "="*70)
print("BFD V26.00 COLUMN SUMMARY BY TYPE")
print("="*70)

total = 0
summary = {}
for cat in sorted(categories.keys()):
    count = len(categories[cat])
    total += count
    summary[cat] = count
    print(f"  {cat}: {count} columns")

print(f"\n  TOTAL: {total} columns")
print("="*70)

# Save full list
output = {
    'summary': summary,
    'total_columns': total,
    'columns_by_type': {k: categories[k] for k in sorted(categories.keys())}
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nFull list saved to: {OUTPUT_PATH}")

# Print all columns
print("\n" + "="*70)
print("COMPLETE COLUMN LIST BY TYPE")
print("="*70)

for cat in sorted(categories.keys()):
    print(f"\n### {cat.upper()} ({len(categories[cat])} columns) ###")
    for col in sorted(categories[cat]):
        print(f"  - {col}")
