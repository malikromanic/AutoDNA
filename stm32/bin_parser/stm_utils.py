import struct
from collections import defaultdict
import numpy as np
from AutoDNA.stm32.bin_parser.packet import Packet, CHUNK_NAMES


def unstuff_bytes(data: bytes) -> bytes:
    result = []
    i = 0
    while i < len(data):
        if data[i] == 0xFE:
            if i + 1 >= len(data):
                raise ValueError("Manjka byte")
            result.append(data[i + 1] ^ 0xFE)
            i += 2
        else:
            result.append(data[i])
            i += 1
    return bytes(result)


def crc16_update(crc: int, byte: int) -> int:
    crc ^= byte
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc >>= 1
    return crc


def crc16_compute(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc = crc16_update(crc, byte)
    return crc


def parse_packet(data: bytes) -> list[Packet] | None:
    if data[0:2] != b'\xFF\xFF':
        return None

    payload = unstuff_bytes(data[2:])
    timestamp = struct.unpack('<I', payload[0:4])[0]

    received_crc = struct.unpack('<H', payload[-2:])[0]
    if received_crc != crc16_compute(payload[:-2]):
        return None

    chunks_data = payload[6:-2]
    pos = 0
    packets = []

    while pos < len(chunks_data):
        chunk_id = chunks_data[pos]
        chunk_size = struct.unpack('<H', chunks_data[pos + 1:pos + 3])[0] + 1
        chunk_data = chunks_data[pos + 4:pos + 4 + chunk_size]

        samples = [
            struct.unpack('<hhh', chunk_data[i:i + 6])
            for i in range(0, chunk_size, 6)
        ]

        packets.append(Packet(
            id=chunk_id,
            ts=float(timestamp),
            data=np.array(samples, dtype=np.int16),
        ))
        pos += 4 + chunk_size

    return packets


def read_packets_from_file(filepath: str) -> list[Packet]:
    with open(filepath, 'rb') as f:
        stream = f.read()

    print(f"File size: {len(stream)} bytes")
    packets = []
    i = 0
    packet_count = 0

    while i < len(stream) - 1:
        if stream[i] == 0xFF and stream[i + 1] == 0xFF:
            j = i + 2
            while j < len(stream) - 1:
                if stream[j] == 0xFF and stream[j + 1] == 0xFF:
                    break
                j += 1

            raw = stream[i:] if j >= len(stream) - 1 else stream[i:j]
            raw_brez_counterja = b'\xFF\xFF' + raw[3:]

            try:
                result = parse_packet(raw_brez_counterja)
                if result:
                    packets.extend(result)
                    packet_count += 1
            except Exception as e:
                print(f"  [!] Paket na offset {i:#08x} spodletel: {e}")
                print(f"      raw[:20] = {raw[:20].hex()}")

            i = j
        else:
            i += 1

    print(f"Uspešno parsiranih paketov: {packet_count}")
    return packets


def save_to_npz(packets: list[Packet], filepath: str) -> None:
    sensor_rows: dict[str, list] = {}

    for p in packets:
        rows = sensor_rows.setdefault(p.sensor, [])
        for x, y, z in p.data:
            rows.append([p.ts, x, y, z])

    arrays = {
        name: np.array(rows, dtype=np.float32)
        for name, rows in sensor_rows.items()
    }

    if not filepath.endswith('.npz'):
        filepath += '.npz'

    np.savez(filepath, **arrays)

    print(f"Shranjeno: {filepath}")
    for name, arr in arrays.items():
        print(f"  {name}: {arr.shape[0]} samplov")


def load_from_npz(filepath: str) -> list[Packet]:
    data = np.load(filepath)
    name_to_id = {v: k for k, v in CHUNK_NAMES.items()}

    packets = []
    for name in data.files:
        arr = data[name]
        chunk_id = name_to_id.get(name, 0x00)

        seen: dict[float, list] = {}
        for row in arr:
            seen.setdefault(float(row[0]), []).append(row[1:].astype(np.int16))

        for ts, samples in seen.items():
            packets.append(Packet(
                id=chunk_id,
                ts=ts,
                data=np.array(samples, dtype=np.int16),
            ))

    packets.sort(key=lambda p: (p.ts, p.id))
    return packets


def group_packets(seznam_paketov: list[Packet]) -> dict:
    ret = defaultdict(list)
    for p in seznam_paketov:
        ret[p.id].append(p)
    return ret