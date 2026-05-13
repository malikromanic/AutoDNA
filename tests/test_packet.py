import numpy as np

from packet import CHUNK_NAMES, Packet


def test_chunk_names_known_ids():
    assert CHUNK_NAMES[0x01] == "gyro"
    assert CHUNK_NAMES[0x02] == "accel"
    assert CHUNK_NAMES[0x03] == "mag"


def test_sensor_name_gyro():
    p = Packet(id=0x01, ts=0.0, data=np.zeros((1, 3), dtype=np.int16))
    assert p.sensor == "gyro"


def test_sensor_name_accel():
    p = Packet(id=0x02, ts=0.0, data=np.zeros((1, 3), dtype=np.int16))
    assert p.sensor == "accel"


def test_sensor_name_mag():
    p = Packet(id=0x03, ts=0.0, data=np.zeros((1, 3), dtype=np.int16))
    assert p.sensor == "mag"


def test_sensor_name_unknown():
    p = Packet(id=0x99, ts=0.0, data=np.zeros((0, 3), dtype=np.int16))
    assert p.sensor == "unknown_0x99"


def test_sample_count():
    data = np.zeros((7, 3), dtype=np.int16)
    p = Packet(id=0x01, ts=0.0, data=data)
    assert p.sample_count == 7


def test_sample_count_empty():
    p = Packet(id=0x01, ts=0.0, data=np.zeros((0, 3), dtype=np.int16))
    assert p.sample_count == 0


def test_repr_contains_sensor_ts_samples():
    p = Packet(id=0x01, ts=1234.0, data=np.zeros((3, 3), dtype=np.int16))
    r = repr(p)
    assert "gyro" in r
    assert "1234" in r
    assert "3" in r
