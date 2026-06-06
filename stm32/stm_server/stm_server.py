"""
stm_server.py — STM32 TCP/IP storitev
======================================
Zagon:  python stm_server.py
Test:   nc 127.0.0.1 5000
"""

import time
import socket
import serial
import serial.tools.list_ports
import threading
import logging
import sys
from pathlib import Path

# ── Nastavitve ─────────────────────────────────────────────────────────────
HOST     = "127.0.0.1"
PORT     = 5000
BAUD     = 9600
TIMEOUT  = 2.0   # sekund tišine = konec prenosa

STM_VID  = 0x0483
STM_PID  = 0x5740
WORK_DIR = Path(__file__).parent

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(WORK_DIR / "stm_server.log", encoding="utf-8"),
    ],
)
log = logging.getLogger()

# ── Stanje ──────────────────────────────────────────────────────────────────
stm_serial = None
stm_port   = None


# ═══════════════════════════════════════════════════════════════════════════
# Serijska komunikacija
# ═══════════════════════════════════════════════════════════════════════════

def is_connected():
    return stm_serial is not None and stm_serial.is_open


def find_stm32():
    for port in serial.tools.list_ports.comports():
        if f"{STM_VID:04X}:{STM_PID:04X}" in port.hwid.upper():
            return port.device
    return None


def send_and_read(command):
    """Pošlje ukaz STM32 in vrne odgovor (bytes)."""
    try:
        stm_serial.write((command + "\r\n").encode())

        response = bytearray()
        last_rx = time.time()
        while True:
            if stm_serial.in_waiting > 0:
                response.extend(stm_serial.read(stm_serial.in_waiting))
                last_rx = time.time()
            elif time.time() - last_rx > TIMEOUT:
                break
        return bytes(response)
    except Exception as e:
        log.warning(f"Serijska napaka: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Ukazi
# ═══════════════════════════════════════════════════════════════════════════

def cmd_status():
    if is_connected():
        return f"STM32 is connected at {stm_port}"
    return "STM32 is not connected"


def cmd_get_file(filename):
    if not is_connected():
        return "FAIL: STM32 is not connected"

    data = send_and_read(f"GET {filename}")
    if not data:
        return f"FAIL: Could not receive {filename}"

    (WORK_DIR / filename).write_bytes(data)
    log.info(f"Shranjeno: {filename} ({len(data)} bytes)")
    return f"File {filename} from STM32 has been processed"


def cmd_get_last():
    if not is_connected():
        return "FAIL: STM32 is not connected"

    response = send_and_read("LIST")
    if not response:
        return "FAIL: No files found on STM32"

    files = [line.split()[0] for line in response.decode(errors="replace").splitlines() if ".BIN" in line.upper()]
    if not files:
        return "FAIL: No files found on STM32"

    result = cmd_get_file(files[-1])
    return "Last file from STM32 has been processed" if "processed" in result else result


def cmd_get_all():
    if not is_connected():
        return "FAIL: STM32 is not connected"

    response = send_and_read("LIST")
    if not response:
        return "FAIL: No files found on STM32"

    files = [line.split()[0] for line in response.decode(errors="replace").splitlines() if ".BIN" in line.upper()]
    if not files:
        return "FAIL: No files found on STM32"

    for filename in files:
        cmd_get_file(filename)
    return "All files from STM32 are processed"


def cmd_delete():
    if not is_connected():
        return "FAIL: STM32 is not connected"

    send_and_read("DELETE")
    return "All files on STM32 are deleted"

def cmd_list():
    if not is_connected():
        return "FAIL: STM32 is not connected"

    response = send_and_read("LIST")
    if not response:
        return "FAIL: No files found on STM32"

    files = [line.strip() for line in response.decode(errors="replace").splitlines() if line.strip()]
    if not files:
        return "No files found on STM32"
    
    return "Files on STM32:\n" + "\n".join(files)


def handle_command(cmd):
    cmd = cmd.strip()
    upper = cmd.upper()

    if upper == "STATUS":
        return cmd_status()
    elif upper == "GET_LAST":
        return cmd_get_last()
    elif upper == "GET_ALL":
        return cmd_get_all()
    elif upper.startswith("GET_FILE|"):
        filename = cmd[len("GET_FILE|"):].strip()
        return cmd_get_file(filename) if filename else "FAIL: Missing filename"
    elif upper == "DELETE":
        return cmd_delete()
    elif upper == "LIST":
        return cmd_list()
    else:
        return f"FAIL: Unknown command '{cmd}'"


# ═══════════════════════════════════════════════════════════════════════════
# STM32 hot-plug (teče v ozadju)
# ═══════════════════════════════════════════════════════════════════════════

def watch_for_stm32():
    global stm_serial, stm_port
    while True:
        time.sleep(2)
        found = find_stm32()

        if found and not is_connected():
            try:
                stm_serial = serial.Serial(port=found, baudrate=BAUD, timeout=1)
                stm_port = found
                log.info(f"STM32 detected at {found}")
            except Exception as e:
                log.warning(f"Ne morem odpreti {found}: {e}")

        elif not found and is_connected():
            try:
                stm_serial.close()
            except Exception:
                pass
            stm_serial = None
            stm_port = None
            log.info("STM32 has disconnected")

        elif is_connected():
            try:
                _ = stm_serial.in_waiting
            except Exception:
                stm_serial = None
                stm_port = None
                log.info("STM32 connection lost")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global stm_serial, stm_port

    log.info("STM32 service se zaganja...")

    # STM32 že priključen ob zagonu?
    found = find_stm32()
    if found:
        try:
            stm_serial = serial.Serial(port=found, baudrate=BAUD, timeout=1)
            stm_port = found
            log.info(f"STM32 že priključen na {found}")
        except Exception as e:
            log.warning(f"Napaka: {e}")

    threading.Thread(target=watch_for_stm32, daemon=True).start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    log.info(f"Poslušam na {HOST}:{PORT} ...")

    try:
        while True:
            client, address = server.accept()
            log.info(f"Odjemalec: {address}")

            # Pozdrav
            if is_connected():
                client.sendall(f"Connected to SPO STM32 service - STM32 at {stm_port}\n".encode())
            else:
                client.sendall(b"Connected to SPO STM32 service - No STM32 detected\n")

            # Sprejemamo ukaze dokler odjemalec ne zapre povezave
            buffer = b""
            try:
                while True:
                    data = client.recv(1024)
                    if not data:
                        break
                    buffer += data
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        cmd = line.replace(b"\r", b"").decode(errors="replace").strip()
                        if not cmd:
                            continue
                        log.info(f"Ukaz: {cmd!r}")
                        response = handle_command(cmd)
                        client.sendall((response + "\n").encode())
            except Exception:
                pass
            finally:
                client.close()
                log.info(f"Odjemalec odklopljen: {address}")

    except KeyboardInterrupt:
        log.info("Ustavitev (Ctrl+C)")
    finally:
        server.close()
        if is_connected():
            stm_serial.close()


if __name__ == "__main__":
    main()