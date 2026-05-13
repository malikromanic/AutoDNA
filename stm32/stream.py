import queue
import threading
from collections import deque
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sis import group_packets
from stm_utils import parse_packet

PORT = 'COM4'
OUTPUT_FILENAME = "sensor.bin"
WINDOW_SECONDS = 5

running = True
data_queue = queue.Queue()

SENSOR_IDS = [1, 2, 3]
SENSOR_NAMES = ["Gyroscope", "Accelerometer", "Magnetometer"]
buffers: dict[int, deque] = {}

fig, graphs = plt.subplots(3, figsize=(10, 8))
lines = []
for graph, name in zip(graphs, SENSOR_NAMES):
    graph.set_title(name)
    graph.set_xlabel("Time (s)")
    sensor_lines = [graph.plot([], [], label=c)[0] for c in ("x", "y", "z")]
    graph.legend(loc="upper right")
    lines.append(sensor_lines)
plt.tight_layout()

sensors_fvz = {
    1: 100,
    2: 25,
    3: 10
}


def serial_processor():
    global running
    buffer = bytearray()
    try:
        ser = serial.Serial(port=PORT, baudrate=9600, timeout=1)
        print(f'Connected ({PORT})')
        with open(OUTPUT_FILENAME, "wb") as file:
            while running:
                if ser.in_waiting > 0:
                    chunk = ser.read(ser.in_waiting)
                    file.write(chunk)
                    buffer.extend(chunk)

                    marker = b'\xFF\xFF'
                    while True:
                        start_idx = buffer.find(marker)
                        if start_idx == -1:
                            break
                        end_idx = buffer.find(marker, start_idx + 1)
                        if end_idx == -1:
                            buffer = buffer[start_idx:]
                            break
                        raw = buffer[start_idx:end_idx]
                        buffer = buffer[end_idx:]
                        no_counter = marker + raw[3:]
                        packets = parse_packet(no_counter)
                        if packets:
                            data_queue.put(packets)
    except (serial.SerialException, serial.SerialTimeoutException,
            KeyboardInterrupt):
        print("Connection closed")
    finally:
        ser.close()
        running = False


def update(_frame):
    while not data_queue.empty():
        packets = data_queue.get_nowait()
        grouped = group_packets(packets)
        for sid, plist in grouped.items():
            if sid not in buffers:
                buffers[sid] = deque(maxlen=sensors_fvz[sid] * WINDOW_SECONDS)
            for p in plist:
                buffers[sid].extend(p.data)

    for i, sid in enumerate(SENSOR_IDS):
        if sid not in buffers or len(buffers[sid]) == 0:
            continue
        arr = np.array(buffers[sid], dtype=float)
        fvz = sensors_fvz[sid]
        t = np.arange(len(arr)) / fvz
        for j, line in enumerate(lines[i]):
            line.set_data(t, arr[:, j])
        graphs[i].set_xlim(0, max(t[-1], WINDOW_SECONDS))
        graphs[i].relim()
        graphs[i].autoscale_view(scalex=False, scaley=True)

    return [line for trio in lines for line in trio]


def graphing():
    thread = threading.Thread(target=serial_processor, daemon=True)
    thread.start()
    ani = FuncAnimation(fig, update, interval=100, blit=True,  # noqa: F841
                        cache_frame_data=False)
    plt.show()
    global running
    running = False


if __name__ == "__main__":
    graphing()