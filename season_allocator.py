#!/usr/bin/env python3
"""
SEASON ALLOCATOR - Standalone for Merge Engine
===============================================
Splits aggregated TV show views into per-season allocations.

Uses linear algebra (NOT ML) to decompose:
    V_total -> V_s1, V_s2, ..., V_sn

Based on:
1. Viewer Behavior Model (continuers, acquired, rewatchers)
2. Genre-specific Temporal Decay
3. Recency Bonus

VERSION: 1.0.0 | ALGO 95.4 Compliant
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SeasonInfo:
    """Information about a single season"""
    season_number: int
    release_date: Optional[datetime] = None
    episode_count: Optional[int] = None
    runtime_minutes: Optional[int] = None


@dataclass
class AllocationResult:
    """Result of season allocation"""
    season_number: int
    allocated_views: int
    allocation_percent: float
    weight_breakdown: Dict[str, float]
    confidence: float


# =============================================================================
# VIEWER BEHAVIOR MODEL
# =============================================================================

class ViewerBehaviorModel:
    """
    Models three viewer types for multi-season shows:

    1. CONTINUERS (45%): Watch latest season on release
    2. ACQUIRED (35%): New viewers, start from S1
    3. REWATCHERS (20%): Rewatch before new season
    """

    DEFAULT_CONTINUER_PCT = 0.45
    DEFAULT_ACQUIRED_PCT = 0.35
    DEFAULT_REWATCHER_PCT = 0.20

    def __init__(self, continuer_pct=None, acquired_pct=None, rewatcher_pct=None):
        self.continuer_pct = continuer_pct or self.DEFAULT_CONTINUER_PCT
        self.acquired_pct = acquired_pct or self.DEFAULT_ACQUIRED_PCT
        self.rewatcher_pct = rewatcher_pct or self.DEFAULT_REWATCHER_PCT

        total = self.continuer_pct + self.acquired_pct + self.rewatcher_pct
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Viewer percentages must sum to 1.0, got {total}")

    def get_continuer_weights(self, num_seasons: int, latest_season: int) -> List[float]:
        """Continuers mostly watch latest season (70% latest, 20% second-latest)"""
        weights = [0.0] * num_seasons

        if num_seasons == 1:
            weights[0] = 1.0
        elif num_seasons == 2:
            weights[latest_season - 1] = 0.85
            weights[latest_season - 2] = 0.15
        else:
            weights[latest_season - 1] = 0.70
            weights[latest_season - 2] = 0.20
            older_count = num_seasons - 2
            if older_count > 0:
                per_old = 0.10 / older_count
                for i in range(older_count):
                    weights[i] = per_old

        return weights

    def get_acquired_weights(self, num_seasons: int, completion_decay: float = 0.7) -> List[float]:
        """Acquired viewers start from S1, decay forward (70% completion per season)"""
        weights = [0.0] * num_seasons
        weights[0] = 1.0

        for i in range(1, num_seasons):
            weights[i] = weights[i-1] * completion_decay

        total = sum(weights)
        return [w / total for w in weights]

    def get_rewatcher_weights(self, num_seasons: int, latest_season: int) -> List[float]:
        """Rewatchers favor S1 (nostalgia 40%) and latest (catch-up 30%)"""
        weights = [0.0] * num_seasons

        if num_seasons == 1:
            weights[0] = 1.0
        elif num_seasons == 2:
            weights[0] = 0.50
            weights[1] = 0.50
        else:
            weights[0] = 0.40
            weights[latest_season - 1] = 0.30
            middle_count = num_seasons - 2
            if middle_count > 0:
                per_middle = 0.30 / middle_count
                for i in range(1, latest_season - 1):
                    weights[i] = per_middle

        return weights

    def get_combined_weights(self, num_seasons: int, latest_season: int) -> List[float]:
        """Combine all viewer types"""
        continuer_w = self.get_continuer_weights(num_seasons, latest_season)
        acquired_w = self.get_acquired_weights(num_seasons)
        rewatcher_w = self.get_rewatcher_weights(num_seasons, latest_season)

        combined = []
        for i in range(num_seasons):
            w = (self.continuer_pct * continuer_w[i] +
                 self.acquired_pct * acquired_w[i] +
                 self.rewatcher_pct * rewatcher_w[i])
            combined.append(w)

        return combined


# =============================================================================
# GENRE DECAY MODEL
# =============================================================================

# Default genre decay parameters (lambda, peak_A, baseline_B)
DEFAULT_GENRE_DECAY = {
    "drama_serial": {"lambda_daily": 0.019, "peak_A": 0.88, "baseline_B": 0.12},
    "comedy": {"lambda_daily": 0.015, "peak_A": 0.85, "baseline_B": 0.15},
    "action": {"lambda_daily": 0.029, "peak_A": 0.92, "baseline_B": 0.08},
    "sci_fi": {"lambda_daily": 0.022, "peak_A": 0.90, "baseline_B": 0.10},
    "horror": {"lambda_daily": 0.037, "peak_A": 0.92, "baseline_B": 0.08},
    "reality_tv": {"lambda_daily": 0.009, "peak_A": 0.75, "baseline_B": 0.25},
    "documentary": {"lambda_daily": 0.012, "peak_A": 0.80, "baseline_B": 0.20},
    "animation": {"lambda_daily": 0.010, "peak_A": 0.82, "baseline_B": 0.18},
    "thriller": {"lambda_daily": 0.025, "peak_A": 0.90, "baseline_B": 0.10},
    "romance": {"lambda_daily": 0.018, "peak_A": 0.85, "baseline_B": 0.15},
}


class GenreDecayModel:
    """Genre-specific temporal decay for view allocation"""

    def __init__(self, weighters_path: str = None):
        self.genres = DEFAULT_GENRE_DECAY.copy()

        # Try to load from weighters file
        if weighters_path:
            decay_file = Path(weighters_path) / "genre decay table.json"
            if decay_file.exists():
                try:
                    with open(decay_file, 'r') as f:
                        data = json.load(f)
                        self.genres.update(data.get("genres", {}))
                except:
                    pass

    def get_decay_factor(self, genre: str, days_since_release: int) -> float:
        """
        Calculate decay factor: T = A * exp(-lambda * d) + B
        """
        genre_key = genre.lower().replace(" ", "_").replace("-", "_")
        params = self.genres.get(genre_key, self.genres.get("drama_serial"))

        lambda_g = params.get("lambda_daily", 0.019)
        A = params.get("peak_A", 0.88)
        B = params.get("baseline_B", 0.12)

        decay = A * np.exp(-lambda_g * days_since_release) + B
        return max(0.05, min(1.5, decay))


# =============================================================================
# SEASON ALLOCATOR ENGINE
# =============================================================================

class SeasonAllocatorEngine:
    """
    Main engine for allocating TV show views to seasons.

    Uses linear algebra to solve:
        V_total -> sum(V_si) where sum = V_total

    Weights based on:
        1. Viewer behavior (continuers, acquired, rewatchers)
        2. Genre-specific temporal decay
        3. Season recency bonus
    """

    def __init__(self, weighters_path: str = None):
        self.behavior_model = ViewerBehaviorModel()
        self.decay_model = GenreDecayModel(weighters_path)
        self.weighters_path = weighters_path

    def allocate_views(self,
                       total_views: int,
                       seasons: List[SeasonInfo],
                       genre: str,
                       reporting_period_end: datetime,
                       show_title: str = None) -> Tuple[List[AllocationResult], Dict]:
        """
        Allocate total views across seasons.

        Returns:
            List[AllocationResult] - one per season
            Dict - metadata with methodology
        """
        num_seasons = len(seasons)

        if num_seasons == 0:
            raise ValueError("Must have at least one season")

        if num_seasons == 1:
            return [AllocationResult(
                season_number=1,
                allocated_views=total_views,
                allocation_percent=100.0,
                weight_breakdown={"single_season": 1.0},
                confidence=0.99
            )], {"methodology": "single_season_allocation"}

        # Sort by season number
        seasons = sorted(seasons, key=lambda s: s.season_number)
        latest_season = max(s.season_number for s in seasons)

        # Step 1: Viewer behavior weights
        behavior_weights = self.behavior_model.get_combined_weights(num_seasons, latest_season)

        # Step 2: Temporal decay weights
        decay_weights = []
        for season in seasons:
            if season.release_date:
                days_since = (reporting_period_end - season.release_date).days
                days_since = max(0, days_since)
            else:
                calculated_days = (latest_season - season.season_number) * 365
                days_since = calculated_days

            decay = self.decay_model.get_decay_factor(genre, days_since)
            decay_weights.append(decay)

        # Step 3: Recency bonus
        recency_weights = []
        for season in seasons:
            if season.season_number == latest_season:
                recency_weights.append(1.5)  # 50% boost
            elif season.season_number == latest_season - 1:
                recency_weights.append(1.1)  # 10% boost
            else:
                recency_weights.append(1.0)

        # Step 4: Combine weights
        combined_weights = []
        for i in range(num_seasons):
            w = behavior_weights[i] * decay_weights[i] * recency_weights[i]
            combined_weights.append(w)

        # Normalize
        total_weight = sum(combined_weights)
        normalized_weights = [w / total_weight for w in combined_weights]

        # Step 5: Allocate views
        results = []
        allocated_total = 0

        for i, season in enumerate(seasons):
            allocation_pct = normalized_weights[i] * 100
            views = int(total_views * normalized_weights[i])
            allocated_total += views

            results.append(AllocationResult(
                season_number=season.season_number,
                allocated_views=views,
                allocation_percent=allocation_pct,
                weight_breakdown={
                    "behavior_weight": behavior_weights[i],
                    "decay_weight": decay_weights[i],
                    "recency_weight": recency_weights[i],
                    "combined_normalized": normalized_weights[i]
                },
                confidence=0.85 if season.release_date else 0.70
            ))

        # Adjust for rounding (add remainder to latest season)
        remainder = total_views - allocated_total
        if remainder != 0:
            results[-1].allocated_views += remainder

        metadata = {
            "methodology": "viewer_behavior_decay_recency_model",
            "total_views_input": total_views,
            "total_views_allocated": sum(r.allocated_views for r in results),
            "num_seasons": num_seasons,
            "genre": genre,
            "reporting_period_end": reporting_period_end.isoformat(),
            "show_title": show_title,
            "validation": {
                "sum_equals_total": sum(r.allocated_views for r in results) == total_views,
                "all_positive": all(r.allocated_views >= 0 for r in results)
            }
        }

        return results, metadata


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_season_allocator(weighters_path: str = None) -> SeasonAllocatorEngine:
    """Create a season allocator instance"""
    return SeasonAllocatorEngine(weighters_path)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    print("Season Allocator v1.0 - Standalone Test")
    print("=" * 60)

    engine = SeasonAllocatorEngine()

    # Test: Wednesday (2 seasons)
    seasons = [
        SeasonInfo(season_number=1, release_date=datetime(2022, 11, 23)),
        SeasonInfo(season_number=2, release_date=datetime(2025, 8, 6))
    ]

    results, metadata = engine.allocate_views(
        total_views=197_600_000,
        seasons=seasons,
        genre="comedy",
        reporting_period_end=datetime(2025, 12, 31),
        show_title="Wednesday"
    )

    print("\nWEDNESDAY - Season Allocation")
    print("-" * 40)
    print(f"Total Views: 197,600,000")

    for r in results:
        print(f"\nSeason {r.season_number}:")
        print(f"  Allocated: {r.allocated_views:,} ({r.allocation_percent:.1f}%)")
        print(f"  Confidence: {r.confidence:.0%}")

    print(f"\nValidation: {metadata['validation']}")
    print("\n[DONE] Season Allocator operational")
