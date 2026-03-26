# AutoDNA

> **AI-powered driving style analysis and its impact on vehicle health**

## Overview

AutoDNA is a university faculty project developed at the Institute of Computer Science (FERI, UM). The core idea is to collect real-time driving data and use AI to analyze how a driver's driving style affects the vehicle — fuel consumption, brake wear, engine load, and overall efficiency. Based on the analysis, the system provides personalized recommendations to help the driver become more economical and vehicle-friendly.

The project targets a real-world problem: as car ownership and fuel costs continue to rise, even small improvements in driving habits can translate to significant savings. AutoDNA makes this accessible through an embedded device and a visualization app.

---

## MVP (Minimal Viable Product)

**Driving style → vehicle impact analysis**

- Collect driving data (IMU + OBD2)
- Analyze how the driving pattern affects the vehicle
- Present findings and actionable recommendations to the driver

---

## Planned Extensions

- Driver identification / profiling by driving style
- Brake wear simulation via BeamNG.tech (licensed)
- Predictive maintenance based on simulated + real data

---

## Hardware

### STM32F411 Discovery (data acquisition)
- **Sensors:** accelerometer, gyroscope, magnetometer
- Captures raw motion and orientation data during driving
- Mounted near the OBD2 port (under dashboard), leverages chassis vibrations
- Connected to Raspberry Pi via **USB**

### OBD2 Reader (ELM327 USB)
- Reads vehicle ECU data: RPM, speed, throttle position, coolant temperature, fuel trim, engine load, misfire counts
- Connected to Raspberry Pi via **USB**

### Raspberry Pi (central hub)
- Acts as the main processing unit
- Receives data from both STM32 and OBD2 over USB
- Runs the data pipeline, AI analysis, and serves the visualization app
- Chosen over direct STM32↔OBD2 UART wiring for simplicity and flexibility

---

## Software Architecture

```
STM32 (USB) ──┐
              ├──► Raspberry Pi ──► Data Pipeline ──► AI Analysis ──► Visualization App
OBD2  (USB) ──┘
```

### Data Pipeline (Python)
- Binary packet parsing (`Packet` class)
- Data storage
- Decompression of received data

### AI Analysis
- Pattern recognition on sensor signals
- Driving style classification
- Impact estimation (fuel, brakes, engine)
- Driver recommendations

### Visualization App
- Display of measurements and analysis results
- Driver recommendations UI

---

## Sensors & Data Sources

| Source   | Data                                                        |
|----------|-------------------------------------------------------------|
| STM32 IMU | Acceleration (3-axis), gyroscope (3-axis), magnetometer (3-axis) |
| OBD2     | RPM, speed, throttle position, coolant temp, fuel trim, engine load, misfire counts |

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
