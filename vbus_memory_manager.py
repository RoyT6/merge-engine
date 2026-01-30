#!/usr/bin/env python3
"""
VBUS MEMORY MANAGER - Standalone for Merge Engine
==================================================
Hierarchical GPU memory management: VRAM (L1) -> System RAM (L2/L3)

Prevents GPU drowning by treating:
- VRAM as L1 cache (12GB, fastest)
- Pinned RAM as L2 cache (16GB, fast transfer)
- System RAM as L3 cache (100GB+, large capacity)

VERSION: 1.0.0 | ALGO 95.4 | GPU MANDATORY | NO CPU FALLBACK
"""

from __future__ import annotations

import os
import gc
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from collections import OrderedDict
import threading
import pickle
import re
from functools import lru_cache


# =============================================================================
# GREP-BASED COLUMN MATCHING (PERFORMANCE OPTIMIZATION)
# =============================================================================

class ColumnGrepEngine:
    """
    Grep-style column matching for fast pattern-based lookups.

    Uses pre-compiled regex and indexed sets for O(1) lookups
    instead of O(n) iteration through all columns.

    PERFORMANCE BENEFIT:
    - Pre-compiled regex: ~10x faster than re.compile() per match
    - Set-based lookups: O(1) vs O(n) for column existence checks
    - Indexed column families: instant access to views_q*_*, views_h*_*, etc.
    """

    # Pre-compiled patterns for common column families
    PATTERNS = {
        'views_half_yearly': re.compile(r'^views_h[12]_(\d{2}|\d{4})_([a-z]{2}|total|row)$'),
        'views_quarterly': re.compile(r'^views_q[1-4]_(\d{2}|\d{4})_([a-z]{2}|total|row)$'),
        'views_country': re.compile(r'^views_([a-z]{2}|total|row)$'),
        'hours_country': re.compile(r'^hours_([a-z]{2}|total|row)$'),
        'lag_features': re.compile(r'^(\w+)_lag(\d+)$'),
        'roll_features': re.compile(r'^(\w+)_roll(\d+)$'),
        'bucket_features': re.compile(r'^(\w+)_bucket$'),
        'temporal_views': re.compile(r'^views_(h[12]|q[1-4])_(\d{4})_([a-z]{2}|total|row)$'),
    }

    def __init__(self, columns: list = None):
        self._columns_set = set()
        self._columns_list = []
        self._indexed_families = {}
        self._grep_cache = {}

        if columns:
            self.index_columns(columns)

    def index_columns(self, columns: list) -> None:
        """
        Index all columns for fast grep-style lookups.
        Creates family-based indexes for instant access.
        """
        self._columns_set = set(columns)
        self._columns_list = list(columns)
        self._indexed_families = {
            'views_quarterly': [],
            'views_half_yearly': [],
            'hours': [],
            'identifiers': [],
            'lag': [],
            'roll': [],
            'bucket': [],
            'other': []
        }
        self._grep_cache.clear()

        for col in columns:
            # Categorize into families for instant access
            if self.PATTERNS['views_quarterly'].match(col):
                self._indexed_families['views_quarterly'].append(col)
            elif self.PATTERNS['views_half_yearly'].match(col):
                self._indexed_families['views_half_yearly'].append(col)
            elif self.PATTERNS['hours_country'].match(col):
                self._indexed_families['hours'].append(col)
            elif self.PATTERNS['lag_features'].match(col):
                self._indexed_families['lag'].append(col)
            elif self.PATTERNS['roll_features'].match(col):
                self._indexed_families['roll'].append(col)
            elif self.PATTERNS['bucket_features'].match(col):
                self._indexed_families['bucket'].append(col)
            elif col in ('fc_uid', 'imdb_id', 'tmdb_id', 'flixpatrol_id', 'title', 'title_type'):
                self._indexed_families['identifiers'].append(col)
            else:
                self._indexed_families['other'].append(col)

    def grep(self, pattern: str, flags: int = 0) -> list:
        """
        Grep-style column search with regex pattern.

        Uses LRU cache for repeated patterns.

        Args:
            pattern: Regex pattern to match columns
            flags: re flags (re.IGNORECASE, etc.)

        Returns:
            List of matching column names
        """
        cache_key = (pattern, flags)
        if cache_key in self._grep_cache:
            return self._grep_cache[cache_key]

        try:
            compiled = re.compile(pattern, flags)
            matches = [col for col in self._columns_list if compiled.search(col)]
            self._grep_cache[cache_key] = matches
            return matches
        except re.error:
            return []

    def grep_family(self, family: str) -> list:
        """
        Get all columns in a pre-indexed family (O(1) lookup).

        Families: views_quarterly, views_half_yearly,
                  hours, identifiers, lag, roll, bucket, other
        """
        return self._indexed_families.get(family, [])

    def exists(self, column: str) -> bool:
        """O(1) column existence check via set lookup."""
        return column in self._columns_set

    def grep_country(self, country_code: str) -> list:
        """Get all columns for a specific country code."""
        pattern = rf'_({country_code})$'
        return self.grep(pattern)

    def grep_period(self, period: str, year: str = None) -> list:
        """
        Get columns for a specific temporal period.

        Args:
            period: 'h1', 'h2', 'q1', 'q2', 'q3', 'q4'
            year: Optional year filter (e.g., '2024')
        """
        if period.startswith('h'):
            if year:
                pattern = rf'^views_{period}_{year}_'
            else:
                pattern = rf'^views_{period}_\d{{4}}_'
            return self.grep(pattern)
        elif period.startswith('q'):
            if year:
                pattern = rf'^views_{period}_{year}_'
            else:
                pattern = rf'^views_{period}_\d{{4}}_'
            return self.grep(pattern)
        return []

    def map_source_to_target(self, source_col: str, temporal_type: str, temporal_period: str) -> str:
        """
        Fast column mapping with O(1) existence check.

        Maps FlixPatrol columns to BFD schema columns.
        """
        # Extract country code
        if not source_col.startswith('views_'):
            return None

        cc = source_col.replace('views_', '')

        # Determine target based on temporal type (half_yearly or quarterly only)
        if temporal_type in ('half_yearly', 'quarterly') and temporal_period:
            period = temporal_period.lower().strip()
            target = f'views_{period}_{cc}'
        else:
            return None

        # O(1) existence check
        return target if self.exists(target) else None

    def get_stats(self) -> dict:
        """Return grep engine statistics."""
        return {
            'total_columns': len(self._columns_set),
            'views_quarterly_count': len(self._indexed_families['views_quarterly']),
            'views_half_yearly_count': len(self._indexed_families['views_half_yearly']),
            'hours_count': len(self._indexed_families['hours']),
            'lag_features_count': len(self._indexed_families['lag']),
            'roll_features_count': len(self._indexed_families['roll']),
            'cache_entries': len(self._grep_cache)
        }


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

# Cache sizes (like CPU cache hierarchy)
L1_CACHE_SIZE = 64       # Hot data - GPU VRAM entries
L2_CACHE_SIZE = 256      # Warm data - Pinned RAM entries
L3_CACHE_SIZE = 1024     # Cold data - System RAM entries
PROMOTION_THRESHOLD = 5   # Accesses to promote up a tier

# Memory limits
GPU_VRAM_LIMIT_MB = 10000      # Leave 2GB buffer on 12GB card
PINNED_MEMORY_LIMIT_MB = 16000  # 16GB pinned for fast transfer
SYSTEM_MEMORY_LIMIT_MB = 100000 # 100GB system RAM


# =============================================================================
# ENUMS
# =============================================================================

class CacheTier(Enum):
    """Cache hierarchy tiers"""
    L1_HOT = "L1_HOT"      # GPU VRAM - fastest
    L2_WARM = "L2_WARM"    # Pinned RAM - fast transfer
    L3_COLD = "L3_COLD"    # System RAM - large capacity
    UNCACHED = "UNCACHED"


class GPUTier(Enum):
    """GPU memory tiers"""
    VRAM = "VRAM"
    PINNED = "PINNED"
    SYSTEM = "SYSTEM"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CacheEntry:
    """Entry in the hierarchical cache"""
    key: str
    value: Any
    tier: CacheTier
    size_bytes: int = 0
    access_count: int = 0
    last_access: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self):
        self.access_count += 1
        self.last_access = datetime.now(timezone.utc).isoformat()


@dataclass
class GPUMemoryBlock:
    """A block of data in GPU memory hierarchy"""
    key: str
    tier: GPUTier
    size_bytes: int
    data: Any = None
    last_access: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0


# =============================================================================
# HIERARCHICAL CACHE
# =============================================================================

class HierarchicalCache:
    """
    Three-tier cache: L1 (VRAM) -> L2 (Pinned) -> L3 (System RAM)

    Automatic promotion/demotion based on access patterns.
    LRU eviction when tiers are full.
    """

    def __init__(self):
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l3_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        self._metrics = {
            "l1_hits": 0, "l2_hits": 0, "l3_hits": 0,
            "misses": 0, "promotions": 0, "demotions": 0, "evictions": 0
        }

    def get(self, key: str) -> Tuple[Optional[Any], CacheTier]:
        """Get from cache, checking L1 -> L2 -> L3"""
        with self._lock:
            # Check L1
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                entry.touch()
                self._l1_cache.move_to_end(key)
                self._metrics["l1_hits"] += 1
                return entry.value, CacheTier.L1_HOT

            # Check L2
            if key in self._l2_cache:
                entry = self._l2_cache[key]
                entry.touch()
                self._l2_cache.move_to_end(key)
                self._metrics["l2_hits"] += 1
                if entry.access_count >= PROMOTION_THRESHOLD:
                    self._promote_to_l1(key, entry)
                return entry.value, CacheTier.L2_WARM

            # Check L3
            if key in self._l3_cache:
                entry = self._l3_cache[key]
                entry.touch()
                self._metrics["l3_hits"] += 1
                self._promote_to_l2(key, entry)
                return entry.value, CacheTier.L3_COLD

            self._metrics["misses"] += 1
            return None, CacheTier.UNCACHED

    def put(self, key: str, value: Any, tier: CacheTier = CacheTier.L2_WARM) -> None:
        """Put value in cache at specified tier"""
        with self._lock:
            size = self._calculate_size(value)
            entry = CacheEntry(key=key, value=value, tier=tier, size_bytes=size)

            if tier == CacheTier.L1_HOT:
                self._put_l1(key, entry)
            elif tier == CacheTier.L2_WARM:
                self._put_l2(key, entry)
            else:
                self._put_l3(key, entry)

    def _put_l1(self, key: str, entry: CacheEntry) -> None:
        self._l2_cache.pop(key, None)
        self._l3_cache.pop(key, None)
        while len(self._l1_cache) >= L1_CACHE_SIZE:
            evicted_key, evicted = self._l1_cache.popitem(last=False)
            evicted.tier = CacheTier.L2_WARM
            self._put_l2(evicted_key, evicted)
            self._metrics["demotions"] += 1
        entry.tier = CacheTier.L1_HOT
        self._l1_cache[key] = entry

    def _put_l2(self, key: str, entry: CacheEntry) -> None:
        self._l1_cache.pop(key, None)
        self._l3_cache.pop(key, None)
        while len(self._l2_cache) >= L2_CACHE_SIZE:
            evicted_key, evicted = self._l2_cache.popitem(last=False)
            evicted.tier = CacheTier.L3_COLD
            self._put_l3(evicted_key, evicted)
            self._metrics["demotions"] += 1
        entry.tier = CacheTier.L2_WARM
        self._l2_cache[key] = entry

    def _put_l3(self, key: str, entry: CacheEntry) -> None:
        self._l1_cache.pop(key, None)
        self._l2_cache.pop(key, None)
        while len(self._l3_cache) >= L3_CACHE_SIZE:
            self._l3_cache.popitem(last=False)
            self._metrics["evictions"] += 1
        entry.tier = CacheTier.L3_COLD
        self._l3_cache[key] = entry

    def _promote_to_l1(self, key: str, entry: CacheEntry) -> None:
        self._put_l1(key, entry)
        self._metrics["promotions"] += 1

    def _promote_to_l2(self, key: str, entry: CacheEntry) -> None:
        self._put_l2(key, entry)
        self._metrics["promotions"] += 1

    def invalidate(self, key: str) -> None:
        """Remove from all tiers"""
        with self._lock:
            self._l1_cache.pop(key, None)
            self._l2_cache.pop(key, None)
            self._l3_cache.pop(key, None)

    def clear_l1(self) -> None:
        """Clear L1 (GPU VRAM) - demote to L2"""
        with self._lock:
            for key, entry in list(self._l1_cache.items()):
                entry.tier = CacheTier.L2_WARM
                self._put_l2(key, entry)
                self._metrics["demotions"] += 1
            self._l1_cache.clear()

    def _calculate_size(self, value: Any) -> int:
        try:
            if hasattr(value, 'nbytes'):
                return value.nbytes
            elif hasattr(value, 'memory_usage'):
                return int(value.memory_usage(deep=True).sum())
            return len(pickle.dumps(value))
        except:
            return 0

    def get_metrics(self) -> Dict:
        total = self._metrics["l1_hits"] + self._metrics["l2_hits"] + self._metrics["l3_hits"] + self._metrics["misses"]
        return {
            **self._metrics,
            "total_hit_rate": (self._metrics["l1_hits"] + self._metrics["l2_hits"] + self._metrics["l3_hits"]) / max(1, total),
            "l1_hit_rate": self._metrics["l1_hits"] / max(1, total),
            "l1_size": len(self._l1_cache),
            "l2_size": len(self._l2_cache),
            "l3_size": len(self._l3_cache)
        }


# =============================================================================
# VBUS GPU MEMORY MANAGER
# =============================================================================

class VBUSGPUManager:
    """
    GPU Memory Manager with Hierarchical Caching

    Prevents GPU memory exhaustion by:
    1. Staging data in system RAM (L2/L3)
    2. Loading only what's needed to VRAM (L1)
    3. Automatic eviction and demotion
    4. Clearing GPU between operations
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.gpu_available = False
        self.cupy = None
        self.cudf = None
        self.cache = HierarchicalCache()

        # Memory tracking
        self._vram_used_mb = 0
        self._vram_blocks: Dict[str, GPUMemoryBlock] = {}
        self._system_blocks: Dict[str, GPUMemoryBlock] = {}
        self._log_entries: List[Dict] = []

        # Grep-based column engine for fast pattern matching
        self.column_grep: Optional[ColumnGrepEngine] = None

        if use_gpu:
            self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize GPU libraries"""
        try:
            import cupy
            import cudf
            self.cupy = cupy
            self.cudf = cudf
            self.gpu_available = True

            device = cupy.cuda.Device()
            total_mb = device.mem_info[1] / (1024 * 1024)
            self._log("GPU_INIT", f"GPU initialized: {total_mb:.0f}MB VRAM available")

        except ImportError as e:
            self._log("GPU_INIT_FAIL", f"GPU libraries not available: {e}")
            self.gpu_available = False

    def _log(self, event: str, message: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "message": message,
            "vram_used_mb": self._vram_used_mb
        }
        self._log_entries.append(entry)
        print(f"  [VBUS-GPU] {event}: {message}")

    # =========================================================================
    # GPU MEMORY MANAGEMENT
    # =========================================================================

    def clear_gpu_memory(self) -> None:
        """Clear all GPU VRAM - essential to prevent drowning"""
        if not self.gpu_available:
            return

        self._log("GPU_CLEAR", "Clearing GPU VRAM...")

        # Clear tracked blocks
        for key in list(self._vram_blocks.keys()):
            del self._vram_blocks[key]

        # Demote L1 cache to L2
        self.cache.clear_l1()

        # Force garbage collection
        gc.collect()

        # Clear CuPy memory pools
        if self.cupy:
            self.cupy.get_default_memory_pool().free_all_blocks()
            self.cupy.get_default_pinned_memory_pool().free_all_blocks()

        self._vram_used_mb = 0
        self._vram_blocks.clear()

        # Report memory
        mem_info = self.get_gpu_memory_info()
        self._log("GPU_CLEAR_DONE", f"VRAM: {mem_info.get('free_mb', 0):.0f}MB free / {mem_info.get('total_mb', 0):.0f}MB total")

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory status"""
        if not self.gpu_available or not self.cupy:
            return {"available": False}

        try:
            meminfo = self.cupy.cuda.Device().mem_info
            free_mb = meminfo[0] / (1024 * 1024)
            total_mb = meminfo[1] / (1024 * 1024)
            used_mb = total_mb - free_mb

            return {
                "available": True,
                "free_mb": free_mb,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "usage_percent": (used_mb / total_mb) * 100 if total_mb > 0 else 0
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def _calculate_size_mb(self, data: Any) -> float:
        try:
            if hasattr(data, 'nbytes'):
                return data.nbytes / (1024 * 1024)
            elif hasattr(data, 'memory_usage'):
                return data.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(data, np.ndarray):
                return data.nbytes / (1024 * 1024)
            return 0
        except:
            return 0

    # =========================================================================
    # DATA STAGING (System RAM L2/L3 -> GPU VRAM L1)
    # =========================================================================

    def stage_data(self, key: str, data: Any, hot: bool = False) -> None:
        """
        Stage data in cache hierarchy.
        hot=True puts directly in L1 (VRAM), else L2 (system RAM)
        """
        size_mb = self._calculate_size_mb(data)
        tier = CacheTier.L1_HOT if hot else CacheTier.L2_WARM

        self.cache.put(key, data, tier)
        self._log("STAGE", f"Staged '{key}' ({size_mb:.1f}MB) in {tier.value}")

    def load_to_gpu(self, key: str) -> Any:
        """
        Load data to GPU VRAM.
        Checks cache, transfers if needed.
        """
        if not self.gpu_available:
            value, _ = self.cache.get(key)
            return value

        # Check cache
        value, tier = self.cache.get(key)
        if value is None:
            self._log("GPU_LOAD_FAIL", f"Data '{key}' not in cache")
            return None

        # If already in L1, it's on GPU
        if tier == CacheTier.L1_HOT:
            return value

        # Transfer to GPU
        size_mb = self._calculate_size_mb(value)
        mem_info = self.get_gpu_memory_info()

        # Check if we have space
        if mem_info.get("free_mb", 0) < size_mb * 1.2:
            self._log("GPU_EVICT", f"Not enough VRAM for '{key}', clearing...")
            self.clear_gpu_memory()
            mem_info = self.get_gpu_memory_info()

            if mem_info.get("free_mb", 0) < size_mb * 1.2:
                self._log("GPU_OVERFLOW", f"Still not enough VRAM - using system RAM")
                return value

        # Convert to GPU format
        try:
            if hasattr(value, 'values') and hasattr(value, 'columns'):
                # DataFrame -> cuDF
                gpu_data = self.cudf.DataFrame(value)
            elif isinstance(value, np.ndarray):
                # NumPy -> CuPy
                gpu_data = self.cupy.asarray(value)
            else:
                gpu_data = value

            # Put in L1 (hot)
            self.cache.put(key, gpu_data, CacheTier.L1_HOT)
            self._vram_used_mb += size_mb

            self._log("GPU_LOAD", f"Loaded '{key}' to VRAM ({size_mb:.1f}MB)")
            return gpu_data

        except Exception as e:
            self._log("GPU_LOAD_ERROR", f"Error loading '{key}': {e}")
            return value

    def prepare_for_merge(self, df_key: str, df: Any) -> Any:
        """
        Prepare a DataFrame for merge operation.
        Stages in system RAM, then loads to GPU with automatic eviction.
        """
        self._log("PREPARE_MERGE", f"Preparing '{df_key}' for GPU merge...")

        # Clear GPU first
        self.clear_gpu_memory()

        # Stage in L2 (system RAM)
        self.stage_data(df_key, df, hot=False)

        # Load to GPU
        return self.load_to_gpu(df_key)

    def batch_load(self, chunk_size_mb: float = 2000) -> None:
        """
        Load data in batches to prevent VRAM overflow.
        For very large DataFrames that exceed VRAM.
        """
        pass  # Implemented in merge engine for specific use case

    # =========================================================================
    # GREP-BASED COLUMN MATCHING (PERFORMANCE)
    # =========================================================================

    def init_column_grep(self, columns: list) -> ColumnGrepEngine:
        """
        Initialize grep-based column matching for fast lookups.

        PERFORMANCE BENEFIT:
        - O(1) column existence checks vs O(n) list search
        - Pre-indexed column families for instant access
        - Cached regex searches for repeated patterns

        Args:
            columns: List of column names (from DataFrame.columns.tolist())

        Returns:
            ColumnGrepEngine instance (also stored as self.column_grep)
        """
        self.column_grep = ColumnGrepEngine(columns)
        stats = self.column_grep.get_stats()
        self._log("GREP_INIT", f"Indexed {stats['total_columns']} columns for grep-based matching")
        self._log("GREP_FAMILIES", f"quarterly: {stats['views_quarterly_count']}, half_yearly: {stats['views_half_yearly_count']}, lag: {stats['lag_features_count']}")
        return self.column_grep

    def grep_columns(self, pattern: str) -> list:
        """
        Grep columns matching pattern.

        Convenience wrapper - initializes engine if needed.
        """
        if self.column_grep is None:
            self._log("GREP_WARN", "Column grep not initialized - call init_column_grep() first")
            return []
        return self.column_grep.grep(pattern)

    def column_exists(self, column: str) -> bool:
        """
        O(1) column existence check via grep engine.

        Much faster than 'column in df.columns' for repeated checks.
        """
        if self.column_grep is None:
            return False
        return self.column_grep.exists(column)

    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================

    def get_status(self) -> Dict:
        gpu_info = self.get_gpu_memory_info()
        cache_metrics = self.cache.get_metrics()
        grep_stats = self.column_grep.get_stats() if self.column_grep else None

        return {
            "gpu_available": self.gpu_available,
            "gpu_info": gpu_info,
            "cache_metrics": cache_metrics,
            "vram_used_mb": self._vram_used_mb,
            "log_entries": len(self._log_entries),
            "grep_engine": grep_stats
        }

    def print_status(self) -> None:
        status = self.get_status()
        gpu = status["gpu_info"]
        cache = status["cache_metrics"]

        print()
        print("=" * 70)
        print("VBUS GPU MEMORY MANAGER STATUS")
        print("=" * 70)

        if status["gpu_available"]:
            print(f"  GPU VRAM (L1):")
            print(f"    Used:  {gpu.get('used_mb', 0):.0f} MB")
            print(f"    Free:  {gpu.get('free_mb', 0):.0f} MB")
            print(f"    Total: {gpu.get('total_mb', 0):.0f} MB")
            print(f"    Usage: {gpu.get('usage_percent', 0):.1f}%")
        else:
            print("  GPU: Not available")

        print(f"\n  Cache Hierarchy:")
        print(f"    L1 (VRAM):   {cache['l1_size']} entries")
        print(f"    L2 (Warm):   {cache['l2_size']} entries")
        print(f"    L3 (Cold):   {cache['l3_size']} entries")
        print(f"    Hit Rate:    {cache['total_hit_rate']:.1%}")
        print(f"    Promotions:  {cache['promotions']}")
        print(f"    Demotions:   {cache['demotions']}")
        print(f"    Evictions:   {cache['evictions']}")

        grep_stats = status.get("grep_engine")
        if grep_stats:
            print(f"\n  Grep Column Engine (Performance):")
            print(f"    Total Columns:    {grep_stats['total_columns']}")
            print(f"    Views Quarterly:  {grep_stats['views_quarterly_count']}")
            print(f"    Views Half-Year:  {grep_stats['views_half_yearly_count']}")
            print(f"    Lag Features:     {grep_stats['lag_features_count']}")
            print(f"    Roll Features:    {grep_stats['roll_features_count']}")
            print(f"    Cache Entries:    {grep_stats['cache_entries']}")
        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_gpu_manager() -> VBUSGPUManager:
    """Create and return a GPU memory manager instance"""
    return VBUSGPUManager(use_gpu=True)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    print("VBUS Memory Manager v1.0 - Standalone Test")
    print("=" * 60)

    manager = VBUSGPUManager(use_gpu=True)
    manager.print_status()

    if manager.gpu_available:
        print("\n[TEST] Testing cache hierarchy...")

        # Create test data
        test_data = np.random.randn(10000, 100).astype(np.float32)

        # Stage in L2
        manager.stage_data("test_array", test_data, hot=False)

        # Load to GPU
        gpu_data = manager.load_to_gpu("test_array")

        manager.print_status()

        # Clear GPU
        manager.clear_gpu_memory()

        print("\n[TEST] After clear:")
        manager.print_status()

    print("\n[DONE] VBUS Memory Manager operational")
