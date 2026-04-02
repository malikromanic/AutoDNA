# AutoDNA

> **AI-powered driving style analysis and its impact on vehicle health**

## Overview

AutoDNA is a university faculty project developed at the Institute of Computer Science (FERI, UM). The core idea is to collect real-time driving data and use AI to analyze how a driver's driving style affects the vehicle — fuel consumption, brake wear, engine load, and overall efficiency. Based on the analysis, the system provides personalized recommendations to help the driver become more economical and vehicle-friendly.

The project targets a real-world problem: as car ownership and fuel costs continue to rise, even small improvements in driving habits can translate to significant savings. AutoDNA makes this accessible through an embedded device and a desktop visualization app.

---

## MVP (Minimal Viable Product)

**Driving style → vehicle impact analysis**

- Collect driving data (IMU via STM32 + OBD2 via ESP32)
- Store all data on SD card during the drive
- After the drive: transfer to laptop, analyze, present findings and recommendations

---

## Planned Extensions

- Real-time streaming from ESP32 to phone/laptop via Bluetooth during the drive
- Driver identification / profiling by driving style
- Brake wear simulation via BeamNG.tech (licensed)
- Predictive maintenance based on simulated + real data

---

## Hardware

### STM32F411 Discovery (central data recorder)
- **Sensors:** accelerometer, gyroscope, magnetometer (onboard IMU)
- Receives OBD2 data from ESP32 via UART
- Stores all data (IMU + OBD2) on SD card (16GB)
- Connected to laptop via USB for post-drive data transfer

### ESP32 (connectivity bridge)
- Receives OBD2 data from ELM327 Bluetooth dongle
- Forwards OBD2 data to STM32 via UART
- Future: streams recorded data from STM32 to phone/laptop via Bluetooth

### ELM327 Bluetooth OBD2 dongle
- Reads vehicle ECU data: RPM, speed, throttle position, coolant temperature, fuel trim, engine load, misfire counts
- Currently (development phase): connects directly to laptop/phone for initial data exploration
- Final plan: communicates exclusively with ESP32

---

## System Architecture

### During the drive
```
OBD2 port
  └──► ELM327 BT dongle ──BT──► ESP32 ──UART──► STM32 ──► SD card
                                                    ▲
                                             IMU sensors
                                         (accel, gyro, mag)
```

### After the drive
```
STM32 SD card ──USB──► Laptop ──► Data Pipeline ──► AI Analysis ──► Visualization App
```

### Future: real-time streaming (planned extension)
```
STM32 ──UART──► ESP32 ──BT/WiFi──► Phone / Laptop ──► Live feedback
```

---

## Software Architecture

### Data Pipeline (PC/Laptop)
- Binary packet parsing
- Merging IMU and OBD2 data streams by timestamp
- Data storage and preprocessing

### AI Analysis
- Pattern recognition on sensor signals
- Driving style classification
- Impact estimation (fuel, brakes, engine)
- Driver recommendations

### Visualization App (desktop)
- Display of measurements and analysis results
- Driver recommendations UI

---

## Sensors & Data Sources

| Source | Data |
|--------|------|
| STM32 IMU | Acceleration (3-axis), gyroscope (3-axis), magnetometer (3-axis) |
| OBD2 (ELM327) | RPM, speed, throttle position, coolant temp, fuel trim, engine load, misfire counts |

---

## University Context

The project spans two semesters and integrates work across multiple courses:

### Summer Semester (Year 2)
| Course | Contribution |
|--------|-------------|
| Signals and Images | Signal acquisition, preprocessing from IMU/OBD2 |
| System Software | PC-side service, USB driver, communication protocol |
| Artificial Intelligence | Pattern recognition, decision system based on measurements |
| Intro to Computer Geometry | 3D CAD/CAM prototype model of the device |

### Winter Semester (Year 3)
| Course | Contribution |
|--------|-------------|
| Computer Graphics | 3D model visualization, graphical display of signals and analysis |
| Multimedia | Compression of sensor data streams |
| Computer & Digital Systems Design | MCU implementation, sensor acquisition, actuator responses |
| Embedded Systems | RTOS integration, PC communication |

---

## Team
3 members — each contributing across all relevant courses.
