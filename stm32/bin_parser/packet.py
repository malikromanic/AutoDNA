import numpy as np

CHUNK_NAMES = {
    0x01: 'gyro',
    0x02: 'accel',
    0x03: 'mag',
}


class Packet:
    id: int
    ts: float
    data: np.ndarray  # shape (M, 3), dtype int16

    def __init__(self, id: int, ts: float, data: np.ndarray):
        self.id = id
        self.ts = ts
        self.data = data

    @property
    def sensor(self) -> str:
        return CHUNK_NAMES.get(self.id, f"unknown_{self.id:#04x}")

    @property
    def sample_count(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (f"Packet(name={self.sensor!r}, ts={self.ts:.0f}ms, "
                f"samples={self.sample_count})")