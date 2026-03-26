# AutoDNA

AI-powered driving style analysis and its impact on vehicle health.

For full project description see [`docs/AutoDNA.md`](docs/AutoDNA.md).

---

## Repository Structure

```
AutoDNA/
├── firmware/          # STM32F411 Discovery code (C/C++)
│   └── src/
├── hub/               # Raspberry Pi — central processing hub
│   ├── pipeline/      # Binary packet parsing and storage
│   ├── obd/           # OBD2 communication
│   └── ai/            # Model inference
├── app/               # Visualization application (desktop)
├── simulation/        # BeamNG.tech integration (planned extension)
├── docs/              # Project documentation
└── README.md
```

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| STM32F411 Discovery | IMU data acquisition (accel, gyro, mag) |
| OBD2 Reader | ELM327 USB — vehicle ECU data |
| Raspberry Pi 5 | Central hub, connects both devices via USB |

---

## Connections

```
STM32F411 (USB) ──┐
                  ├──► Raspberry Pi 5
OBD2 ELM327 (USB)─┘
```

Both devices connect to the Raspberry Pi over USB. No direct STM32↔OBD2 wiring.

---

## Development Setup

### Prerequisites

- Python 3.10+
- STM32CubeIDE (for firmware)
- Git

### Clone the repository

```bash
git clone https://github.com/<org>/AutoDNA.git
cd AutoDNA
```

### Raspberry Pi — hub setup

```bash
cd hub
pip install -r requirements.txt
```

### Running the pipeline

```bash
python hub/pipeline/main.py
```

---

## Team

3 members — FERI, University of Maribor  
Institute of Computer Science — https://cs.feri.um.si/
