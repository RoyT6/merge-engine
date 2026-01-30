# Merge Engine - Standalone GPU-Accelerated Data Pipeline

## Version: 27.00 | ALGO 95.4 Compliant

---

## Overview

The Merge Engine merges FlixPatrol views data into the BFD master database using GPU-accelerated processing (cuDF/RAPIDS). This is a **standalone** directory containing all required components.

### Key Features

- **GPU MANDATORY** - No CPU fallback (ALGO 95.4 compliance)
- **FlixPatrol Senior Credibility** - OVERWRITES existing views data
- **VBUS Memory Management** - Prevents GPU drowning (VRAM as L1 cache)
- **Intelligent View Allocation** - Direct mapping is SENIOR, season allocation only when needed

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `intelligent_merge_engine_v27_gpu.py` | **MAIN** - GPU-accelerated merge engine |
| `vbus_memory_manager.py` | VBUS hierarchical cache (L1/L2/L3) for GPU memory |
| `season_allocator.py` | ALGO2 season allocation engine |
| `run_gpu.sh` | GPU runner script (use in WSL) |

---

## How to Run

```bash
# From WSL:
cd "/mnt/c/Users/RoyT6/Downloads/Merge Engine"
chmod +x run_gpu.sh
./run_gpu.sh

# Or run directly:
python3 intelligent_merge_engine_v27_gpu.py
```

---

## Input Files

| File | Location | Purpose |
|------|----------|---------|
| `BFD_V26.00.parquet` | Downloads/ | Base database |
| `FlixPatrol_Views_v40.10.csv` | Downloads/ | Views by country |
| `FlixPatrol_Multi_Temporal_v41.00.csv` | Downloads/ | Multi-temporal views |
| `FlixPatrol_Netflix_Views_v40.10.csv` | Downloads/ | Netflix views |

---

## Output Files

| File | Location | Purpose |
|------|----------|---------|
| `BFD_V27.00.parquet` | Downloads/ | Merged master database |
| `BFD_V27.00.merge_report.json` | Downloads/ | Detailed merge statistics |

---

## Intelligent View Allocation

### Priority System (SENIOR)

1. **If views MAP directly** (FlixPatrol has season data, or movie, or single-season show):
   - **Direct overwrite** - No calculation needed
   - This is the SENIOR/superior solution

2. **If views DON'T map** (TV show with show-level views, multiple BFD seasons):
   - Use ALGO2 season allocator
   - Apply 26+ weighting components

### Viewer Behavior Model

| Viewer Type | Percentage | Behavior |
|-------------|------------|----------|
| Continuers | 45% | Watch latest season on release |
| Acquired | 35% | New viewers, start from S1 |
| Rewatchers | 20% | Rewatch before new season |

---

## VBUS Memory Management

Prevents GPU "drowning" by staging data across memory tiers:

| Tier | Memory Type | Purpose |
|------|-------------|---------|
| L1 | GPU VRAM | Hot data - active GPU operations |
| L2 | Pinned RAM | Warm data - fast GPU transfer |
| L3 | System RAM | Cold data - large capacity staging |

**RTX 3080 Ti Config:**
- VRAM Limit: 10GB (leaves 2GB buffer)
- L1 Cache: 64 entries
- L2 Cache: 256 entries
- L3 Cache: 1024 entries

---

## Dependencies

### Required (via RAPIDS)
- cuDF
- cuPy
- NumPy
- Pandas

### Weighters (from Downloads/Weighters/)
- Streaming Platform Components (8 files)
- Country Weighting Components (8 files)
- Genre decay, ML features, etc. (18+ files)

---

## Schema V27 Compliance

### Column Naming
- `views_{period}_{year}_{cc}` - Temporal views by country (22 countries)
- `hours_{cc}` - Country hours (22 countries)

### Rule 14 Compliance
Only 3 FlixPatrol columns allowed:
- `flixpatrol_id`
- `flixpatrol_points`
- `flixpatrol_rank`

### 22 Countries
`us, cn, in, gb, br, de, jp, fr, ca, mx, au, es, it, kr, nl, se, sg, hk, ie, ru, tr, row`

---

## Merge Behavior

### FlixPatrol has SENIOR CREDIBILITY
- **OVERWRITES** existing views data
- **FILLS** NULL values
- **APPENDS** new rows for titles not in BFD
- **NEVER DELETES** records

---

## ALGO 95.4 Constraints

- GPU execution is **MANDATORY**
- CPU fallback is **FORBIDDEN**
- All scripts must use cuDF for parquet I/O
- RTX 3080 Ti (12GB VRAM) available via WSL

---

**Last Updated**: 2026-01-23
**Engine Version**: V27.00 (Standalone)
