"""Decode the STM32 data-logger binary format into :class:`~AutoDNA.stm32.bin_parser.packet.Packet` objects.

Wire format: packets are framed by ``0xFF 0xFF``; the payload is byte-stuffed
(``0xFE`` escape) and protected by a trailing CRC16 (Modbus, poly ``0xA001``).
Each payload carries a 32-bit millisecond timestamp and one or more sensor chunks
of int16 XYZ samples. This module parses that format and converts to/from
``.npz`` for downstream processing.
"""

import struct
from collections import defaultdict
import numpy as np
from AutoDNA.stm32.bin_parser.packet import Packet, CHUNK_NAMES


def unstuff_bytes(data: bytes) -> bytes:
    """Reverse the ``0xFE`` byte-stuffing.

    Each ``0xFE`` marks an escaped byte; the following byte is XORed with
    ``0xFE`` to recover the original. Raises :class:`ValueError` if the data ends
    on a dangling escape.
    """
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
    """Fold one byte into a running CRC16 (Modbus, polynomial ``0xA001``)."""
    crc ^= byte
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc >>= 1
    return crc


def crc16_compute(data: bytes) -> int:
    """CRC16 (Modbus) over ``data``, starting from the initial value ``0xFFFF``."""
    crc = 0xFFFF
    for byte in data:
        crc = crc16_update(crc, byte)
    return crc


def parse_packet(data: bytes) -> list[Packet] | None:
    """Parse one framed packet into a list of :class:`Packet`.

    :param data: Bytes starting at the ``0xFF 0xFF`` frame header.
    :returns: One :class:`Packet` per sensor chunk in the frame, or ``None`` if
        the header is missing or the CRC check fails.
    """
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
    """Read a whole ``.BIN`` recording and return every parsed :class:`Packet`.

    Scans for ``0xFF 0xFF`` frame markers and parses each frame; frames that fail
    to parse (bad CRC, truncation) are skipped rather than aborting the read.
    """
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
    """Save packets to a ``.npz``: one array per sensor with columns ``[ts, x, y, z]``.

    A ``.npz`` extension is appended if missing.
    """
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
    """Inverse of :func:`save_to_npz`: load a ``.npz`` back into :class:`Packet`
    objects, sorted by ``(ts, id)``."""
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
    """Group packets by their chunk ``id`` into a ``{id: [Packet, ...]}`` dict."""
    ret = defaultdict(list)
    for p in seznam_paketov:
        ret[p.id].append(p)
    return ret
