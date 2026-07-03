# AutoDNA

AI-powered driving style analysis and its impact on vehicle health.

API reference docs (Sphinx) live in [`docs/`](docs/) — see the "Documentation" section below to build the HTML site or a single PDF.

---

## Repository Structure

```
AutoDNA/
├── firmware/          # STM32F411 Discovery code (C/C++)
│   └── src/
├── esp32/             # ESP32 code — OBD2 BT receiver, UART bridge
│   └── src/
├── pipeline/          # PC-side data pipeline (Python)
├── ai/                # AI analysis and model inference (Python)
├── app/               # Visualization application (desktop)
├── simulation/        # BeamNG.tech integration (planned extension)
├── docs/              # Project documentation
└── README.md
```

---

## Documentation

API reference docs are generated with [Sphinx](https://www.sphinx-doc.org/) from
the docstrings in `app/`, `ai/`, and `stm32/`. The source lives in `docs/source/`.

```bash
pip install sphinx        # one-time
cd docs

# Windows
.\make.bat html           # HTML  -> docs/build/html/index.html
.\make.bat latexpdf       # PDF   -> docs/build/latex/AutoDNA.pdf

# macOS / Linux
make html
make latexpdf
```

- `latexpdf` also needs a LaTeX distribution (e.g. MiKTeX on Windows, TeX Live elsewhere).
- Build from the environment where **PyQt6** is installed (the one used to run the
  app), otherwise the `app/` pages fail to import while `ai`/`stm32`/`tests` still build.

---

## Hardware Requirements

| Component | Role |
|-----------|------|
| STM32F411 Discovery | IMU data acquisition, SD card storage |
| ESP32 | OBD2 Bluetooth receiver, UART bridge to STM32 |
| ELM327 Bluetooth dongle | OBD2 vehicle ECU data |
| SD card (16GB) | On-device data storage during drive |

---

## System Architecture

### During the drive
```
OBD2 port
  └──► ELM327 BT ──BT──► ESP32 ──UART──► STM32 ──► SD card
                                              ▲
                                       IMU sensors
```

### After the drive
```
STM32 SD card ──USB──► Laptop ──► pipeline/ ──► ai/ ──► app/
```

---

## Development Setup

### Prerequisites

- Python 3.10+
- STM32CubeIDE (for STM32 firmware)
- Arduino IDE or ESP-IDF (for ESP32)
- Git

### Clone the repository

```bash
git clone https://github.com/<org>/AutoDNA.git
cd AutoDNA
```

### PC pipeline setup

```bash
cd pipeline
pip install -r requirements.txt
```

### Running the pipeline

```bash
python pipeline/main.py
```

---

## Team

3 members — FERI, University of Maribor  
Institute of Computer Science — https://cs.feri.um.si/
