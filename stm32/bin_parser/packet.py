"""Data model for one decoded STM32 sensor packet.

A packet holds one sensor's samples for a single timestamp. The STM32 groups
samples into *chunks* identified by a small integer id; :data:`CHUNK_NAMES` maps
those ids to human-readable sensor names.
"""

import numpy as np

#: Maps a chunk id (from the binary stream) to its sensor name.
CHUNK_NAMES = {
    0x01: 'gyro',
    0x02: 'accel',
    0x03: 'mag',
}


class Packet:
    """One sensor's samples at a single timestamp.

    :ivar id: Chunk id identifying the sensor (see :data:`CHUNK_NAMES`).
    :ivar ts: Timestamp in milliseconds since the recording started.
    :ivar data: ``(M, 3)`` int16 array of ``M`` XYZ samples.
    """

    id: int
    ts: float
    data: np.ndarray  # shape (M, 3), dtype int16

    def __init__(self, id: int, ts: float, data: np.ndarray):
        """Store the chunk *id*, timestamp *ts* (ms), and ``(M, 3)`` sample array."""
        self.id = id
        self.ts = ts
        self.data = data

    @property
    def sensor(self) -> str:
        """Human-readable sensor name, or ``"unknown_0x.."`` for an unmapped id."""
        return CHUNK_NAMES.get(self.id, f"unknown_{self.id:#04x}")

    @property
    def sample_count(self) -> int:
        """Number of XYZ samples in this packet (rows of :attr:`data`)."""
        return len(self.data)

    def __repr__(self) -> str:
        """Return a concise ``Packet(name=..., ts=..., samples=...)`` string."""
        return (f"Packet(name={self.sensor!r}, ts={self.ts:.0f}ms, "
                f"samples={self.sample_count})")
