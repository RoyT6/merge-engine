# Merge Engine - Session Memory
## ALGO 95.66 Compliant | Schema V27.00 | BFD V27.66

---

## P0 CANON: Studio vs Production Company Classification (2026-01-25)

**CRITICAL RULE:** Classification is by PRIMARY FUNCTION, NOT by fixed lookup list.

### Studios (Creative/On-Set)
| Roles | Activities |
|-------|------------|
| Directors, Writers | Filming, Blocking |
| Cinematographers, Cameramen | Location, Direction |
| Set Designers, Crew | Content Creation, Capture |
| Actors, Casting Directors | Casting, Extras |

### Production Companies (Business/Post)
| Roles | Activities |
|-------|------------|
| Lawyers, Finance | Financing, Legal |
| Post-Production, Final Cut | Editing, Mixing |
| Producers, Editors | Distribution, Marketing |
| Music, Sound, Executives | Promotion, Post-Production |

**Ingestion Rules:**
- P0: Classify by FUNCTION, not name matching
- P1: Accept ANY entity meeting criteria (no lookup required)
- P2: Metadata columns OPTIONAL (NULL acceptable)

**Rule File:** `Schema/RULES/STUDIO_VS_PRODUCTION_COMPANY_RULES.json`

---

## SESSION LOG: 2026-01-24 - V28.00 Star Hierarchy Update

### Star Hierarchy Columns Added

**New Columns in BFD_V26.02:**
| Column | Description | Coverage |
|--------|-------------|----------|
| `star_1` | Top-billed lead actor | 149,073 rows (19.5%) |
| `star_2` | Second-billed actor | 140,557 rows (18.4%) |
| `star_3` | Third-billed actor | 132,256 rows (17.3%) |
| `supporting_cast` | Remaining cast (semicolon-separated) | 123,854 rows (16.2%) |

**Data Source:** `Freckles/00_ACTIVE_SOURCES/TMDB/checkpoints/checkpoint_150000.parquet` (134,149 cast records)

**Unique Titles Covered:** 93,103 (15.3% of 608,628 total)

### Schema Path Updated
- **From:** SCHEMA_V27.00.json
- **To:** SCHEMA_V28.00.json

### Input Database Updated
- **From:** BFD_V26.00.parquet
- **To:** BFD_V26.02.parquet (with star hierarchy columns)

---

## SESSION LOG: 2026-01-23

### What Was Built

**Standalone Merge Engine Directory** with all components self-contained:

1. **intelligent_merge_engine_v27_gpu.py** - Main GPU-accelerated merge engine
   - FlixPatrol OVERWRITES existing views (senior credibility)
   - VBUS memory management integrated
   - Intelligent view allocation (direct mapping SENIOR)
   - Schema V27 compliant column naming

2. **vbus_memory_manager.py** - Standalone VBUS memory management
   - L1 Cache: GPU VRAM (64 entries)
   - L2 Cache: Pinned RAM (256 entries)
   - L3 Cache: System RAM (1024 entries)
   - Prevents GPU drowning on large datasets

3. **season_allocator.py** - Standalone ALGO2 season allocator
   - Viewer behavior model (45% continuers, 35% acquired, 20% rewatchers)
   - Genre-specific temporal decay
   - Recency bonus for latest seasons

4. **run_gpu.sh** - GPU runner script for WSL execution

5. **README.md** - Documentation

---

## Key Design Decisions

### 1. FlixPatrol Senior Credibility
- FlixPatrol data **OVERWRITES** existing views
- This is intentional - FlixPatrol is the authoritative source for views
- No record deletions allowed

### 2. Intelligent View Allocation Priority
```
IF views MAP directly (season data exists, movie, single season):
    → DIRECT OVERWRITE (SENIOR solution)
    → No calculation needed

ELSE IF views DON'T map (TV show, no season, multiple BFD seasons):
    → Use ALGO2 season allocator
    → Apply weighting components
```

### 3. VBUS Memory Management
- VRAM treated as L1 cache (hot data)
- System RAM as L2/L3 (staging)
- Prevents "drowning" the GPU on large datasets

### 4. Schema V27 Compliance
- Column naming: `views_{period}_{year}_{cc}`, `hours_{cc}`
- Rule 14: Only 3 flixpatrol_ columns allowed
- 22 countries tracked

---

## Weighters Used

### Streaming Platform Components (8 files)
- component_streaming_platform_exclusivity_patterns.json
- component_streaming_platform_allocation_weights.json
- component_streaming_financial_data_Q4_2025.json
- component_streaming_platform_international_content_exclusivity.json
- component studio weighting.json
- worlds_streaming_platforms_by_country.json
- netflix_non_global_alternatives.json
- component_trending.json

### Country Weighting Components (8 files)
- 01_country_weights.json
- 02_regional_aggregates.json
- 03_platform_multipliers.json
- 04_temporal_modifiers.json
- 05_content_origin_adjustments.json
- 06_platform_financials_Q4_2025.json
- 07_platform_availability.json
- 08_market_data.json

### Other Weighters (18+ files)
- genre decay table.json
- country_views_weighting_lookup_v1.01.json
- And 16+ more...

---

## Run Commands

```bash
# WSL execution
cd "/mnt/c/Users/RoyT6/Downloads/Merge Engine"
./run_gpu.sh

# Direct execution
python3 intelligent_merge_engine_v27_gpu.py
```

---

## Input/Output

### Input Files
- BFD_V26.00.parquet (base)
- FlixPatrol_Views_v40.10.csv
- FlixPatrol_Multi_Temporal_v41.00.csv
- FlixPatrol_Netflix_Views_v40.10.csv

### Output Files
- BFD_V27.00.parquet (merged master)
- BFD_V27.00.merge_report.json (statistics)

---

## ALGO 95.4 Constraints

- GPU execution is **MANDATORY**
- CPU fallback is **FORBIDDEN**
- cuDF required for parquet I/O
- RTX 3080 Ti (12GB VRAM) via WSL

---

## SESSION LOG: 2026-01-23 (Evening) - BANNED WORDS Enforcement

### Critical Rule Added: BANNED WORDS

**The following words are BANNED from ALL column names, code, and documents:**
- `estimate` / `estimated`
- `lifetime`

These words cannot appear in:
- Column names in BFD or any database
- Variable names in code
- Method names in code
- Documentation text

### Files Modified

| File | Change |
|------|--------|
| `vbus_memory_manager.py` | Renamed `_estimate_size` to `_calculate_size`, `_estimate_size_mb` to `_calculate_size_mb` |
| `season_allocator.py` | Renamed `estimated_days` variable to `calculated_days` |
| `intelligent_merge_engine_v27_gpu.py` | Removed `views_estimated` from comments, added BANNED WORDS note |
| `SCHEMA_V27.00.json` | Removed `lifetime` from temporal_period_types enum, renamed `estimated_total` to `total_column_count`, renamed `row_count_estimate` to `row_count_approx`, `total_columns_estimate` to `total_columns_approx` |
| `Schema v27 updates .txt` | Changed "Estimated Total Columns" to "Approximate Total Columns" |

### Schema V27 Temporal Period Types (Updated)

Valid temporal_period_type values (lifetime and weekly REMOVED):
- `half_yearly`
- `quarterly`
- `monthly`
- `daily`

### Key Clarification: Rows vs Columns

**IMPORTANT**: The BANNED WORDS rule applies to COLUMN NAMES only, NOT to row data.
- Rows with `temporal_period_type='lifetime'` are TITLES, not to be deleted
- The merge engine validates column creation, not row deletion
- 125,445 FlixPatrol rows have lifetime temporal data - these rows are preserved

### Strict Schema Enforcement (Already in Place)

```python
FORBIDDEN_COLUMN_WORDS = {'estimate', 'estimated', 'lifetime'}

def _check_column_exists(self, column_name: str) -> bool:
    col_lower = column_name.lower()
    for forbidden in FORBIDDEN_COLUMN_WORDS:
        if forbidden in col_lower:
            print(f"[FORBIDDEN] Column '{column_name}' contains forbidden word - REJECTED")
            return False
    # ... rest of validation
```

### Pending Work

1. **Generate complete BFD V26 column list by type** - Script created at `list_bfd_columns.py`
2. **Decide handling for 125,445 FlixPatrol rows with lifetime temporal_period_type** - these are valid title rows that need mapping to schema-compliant column patterns
3. **Run full merge engine test** once column mapping strategy is confirmed

---

## SESSION LOG: 2026-01-28 - Date Columns Added to BFD_VIEWS

### Task Completed

Added date columns (`start_year`, `premiere_date`, `finale_date`) to BFD_VIEWS_V27.75.parquet and deleted pre-2000 records.

### Data Sources Used for Date Columns

| Source | Records | Columns Extracted |
|--------|---------|-------------------|
| TMDB_TV_22_MAY24.csv | 164,705 | first_air_date, last_air_date |
| TMDB Movies_729_OCT25.csv | 538,817 | release_date |
| imdb_top_5000_tv_shows.csv | 5,000 | startYear |
| more titles.csv | 5,030 | release_year |

**Location:** `Freckles/00_ACTIVE_SOURCES/`

### Column Mapping

| New Column | Source Priority |
|------------|-----------------|
| `start_year` | Extract year from premiere_date > IMDB startYear > more titles release_year |
| `premiere_date` | TMDB TV first_air_date > TMDB Movies release_date |
| `finale_date` | TMDB TV last_air_date > premiere_date (for movies) |

### Final Database State

| Metric | Value |
|--------|-------|
| Total Records | 559,872 |
| start_year populated | 295,583 (52.8%) |
| start_year NULL | 264,289 (47.2%) - preserved |
| Records pre-2000 | 0 (deleted 205,988) |
| Columns | 1,711 |

### Critical Filter Logic

**CORRECT:** Only delete where `start_year IS NOT NULL AND start_year < 2000`
```python
pre_2000_mask = (df['start_year'].notna()) & (df['start_year'] < 2000)
df_filtered = df[~pre_2000_mask]  # Keeps NULL start_year records
```

**WRONG (causes data loss):**
```python
df_filtered = df[~(df['start_year'] < 2000)]  # Deletes NULL records too!
```

### Source File for Rebuild

If BFD_VIEWS needs rebuilding, use: `Cloudflare/data/BFD_V22.02.parquet` (765,860 records, 1711 columns)

---

**Last Updated**: 2026-01-28
**Version**: Standalone V27.00
