import serial
import sys
import time

PORT     = "COM4"
BAUD     = 9600
TIMEOUT  = 2.0  # seconds of silence = done
FILENAME = sys.argv[1] if len(sys.argv) > 1 else "output.bin"

print(f"Connecting to {PORT}...")
ser = serial.Serial(port=PORT, baudrate=BAUD, timeout=1)
print(f"Connected. Sending GET {FILENAME}...")
ser.write(f"GET {FILENAME}\r\n".encode())

try:
    with open(FILENAME, "wb") as f:
        last_rx = time.time()
        while True:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                f.write(chunk)
                last_rx = time.time()
                print(f"\rReceived {f.tell()} bytes...", end="", flush=True)
            elif time.time() - last_rx > TIMEOUT:
                print(f"\nDone. Saved to '{FILENAME}'")
                break
finally:
    ser.close()