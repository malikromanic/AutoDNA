"""Online DEM elevation lookup used for hill/grade detection.

Hills are derived from ground elevation along the GPS track, not from the
device's own (often inaccurate or absent) GPS altitude reading. Elevation
instead comes from a Digital Elevation Model (DEM) — the same idea
Strava/Garmin use for "elevation correction".

Two key-free public DEM APIs are chained: OpenTopoData EU-DEM 25 m first,
falling back to Open-Meteo's Copernicus GLO-90 (global coverage, ~90 m)
for points outside Europe or when OpenTopoData is unreachable. Results
are cached on disk (:data:`_CACHE_FILE`) so re-opening a drive is instant
and works offline afterwards.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Repo root (…/AutoDNA) — cache lives next to autodna_records.json
_ROOT = Path(__file__).resolve().parent.parent
_CACHE_FILE = _ROOT / "autodna_elev_cache.json"

# Round lat/lon to ~1 m before using as a cache key (6 decimals ≈ 0.11 m).
_CACHE_DECIMALS = 6


class ElevationProvider(ABC):
    """Look up ground elevation (metres) for arrays of lat/lon points."""

    name: str = "elevation"

    @abstractmethod
    def elevations(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Return (N,) float64 elevations in metres; np.nan where unavailable."""
        raise NotImplementedError


# ── On-disk cache ───────────────────────────────────────────────────────────
class _ElevationCache:
    """JSON-backed lat/lon → elevation cache, keyed to 6 decimal places (~0.1 m)."""

    def __init__(self, path: Path = _CACHE_FILE):
        self.path = path
        self._data: dict[str, float] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    @staticmethod
    def _key(lat: float, lon: float) -> str:
        return f"{round(lat, _CACHE_DECIMALS)},{round(lon, _CACHE_DECIMALS)}"

    def get(self, lat: float, lon: float):
        return self._data.get(self._key(lat, lon))

    def put(self, lat: float, lon: float, elev: float):
        self._data[self._key(lat, lon)] = float(elev)

    def flush(self):
        try:
            self.path.write_text(json.dumps(self._data))
        except OSError:
            pass


class OnlineElevationProvider(ElevationProvider):
    """
    Key-free public DEM lookup with caching, batching and a fallback chain.

    Primary : OpenTopoData `eudem25m` (Europe, 25 m).
    Fallback: Open-Meteo elevation (Copernicus DEM GLO-90, ~90 m, global).
    """

    name = "EU-DEM 25 m (OpenTopoData)"

    _OPENTOPO = "https://api.opentopodata.org/v1/{dataset}"
    _OPENMETEO = "https://api.open-meteo.com/v1/elevation"
    _BATCH = 100          # both APIs accept up to 100 points per request
    _TIMEOUT = 20         # seconds
    _SLEEP = 1.05         # OpenTopoData public limit is 1 call/sec

    def __init__(self, dataset: str = "eudem25m"):
        self.dataset = dataset
        self._cache = _ElevationCache()

    # -- public ---------------------------------------------------------------
    def elevations(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Look up elevation for each point, serving cached hits first.

        Misses are fetched from the API in batches of ``_BATCH`` points,
        rate-limited by ``_SLEEP`` between batches, and written back to
        the cache as they resolve.

        Args:
            lats: GPS latitudes.
            lons: GPS longitudes, same length as ``lats``.

        Returns:
            np.ndarray: Elevation in metres per point; ``np.nan`` where
            the lookup failed for both providers.
        """
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        out = np.full(len(lats), np.nan, dtype=float)

        # 1. Serve whatever we can from cache; collect the misses.
        miss_idx = []
        for i in range(len(lats)):
            c = self._cache.get(lats[i], lons[i])
            if c is not None:
                out[i] = c
            else:
                miss_idx.append(i)

        # 2. Fetch misses in batches.
        for b in range(0, len(miss_idx), self._BATCH):
            chunk = miss_idx[b:b + self._BATCH]
            blat = [lats[i] for i in chunk]
            blon = [lons[i] for i in chunk]
            elevs = self._fetch_batch(blat, blon)
            if elevs is None:
                continue  # leave NaN; caller decides how to degrade
            for i, e in zip(chunk, elevs):
                if e is not None and not (isinstance(e, float) and math.isnan(e)):
                    out[i] = e
                    self._cache.put(lats[i], lons[i], e)
            if b + self._BATCH < len(miss_idx):
                time.sleep(self._SLEEP)

        if miss_idx:
            self._cache.flush()
        return out

    # -- internal -------------------------------------------------------------
    def _fetch_batch(self, lats, lons):
        elevs = self._fetch_opentopo(lats, lons)
        if elevs is None:
            self.name = "Copernicus GLO-90 (Open-Meteo)"
            elevs = self._fetch_openmeteo(lats, lons)
        return elevs

    def _fetch_opentopo(self, lats, lons):
        locs = "|".join(f"{a:.6f},{o:.6f}" for a, o in zip(lats, lons))
        url = self._OPENTOPO.format(dataset=self.dataset) + "?" + urllib.parse.urlencode(
            {"locations": locs, "interpolation": "bilinear"}
        )
        try:
            data = self._get_json(url)
            return [r.get("elevation") for r in data["results"]]
        except Exception:
            return None

    def _fetch_openmeteo(self, lats, lons):
        url = self._OPENMETEO + "?" + urllib.parse.urlencode({
            "latitude": ",".join(f"{a:.6f}" for a in lats),
            "longitude": ",".join(f"{o:.6f}" for o in lons),
        })
        try:
            data = self._get_json(url)
            return list(data["elevation"])
        except Exception:
            return None

    def _get_json(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": "AutoDNA/1.0"})
        with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))


def get_default_provider() -> ElevationProvider:
    """Return the key-free online DEM provider."""
    return OnlineElevationProvider()
