# ⚽ Human vs Robot Football (STEM Exhibition)

This repository contains the codebase for the **2v2 Robot Football** exhibit developed for the King's STEM outreach event.

The project created an interactive competitive environment where members of the public piloted **2 Human-Controlled Robots** to compete against **2 Autonomous AI Goalkeepers**.

## System Overview

The exhibit operated on a centralized, low-latency architecture designed to blend autonomous game logic with real-time human teleoperation. The system was divided into three core subsystems:

### 1. Global Vision & Tracking
The match was monitored by a central **AI Server** connected to an overhead webcam. This system functioned as the brains for the AI bots.
* **AprilTag Tracking:** The court was calibrated using static tags on the field corners. Each robot featured a unique AprilTag on its chassis, allowing the server to calculate real-time pose (X, Y) and orientation at high framerates.
* **Game Logic:** The server processed this positional data to track the ball, detect goals, and calculate tracking vectors for the autonomous bots.

### 2. The Control Plane (MQTT)
All communication was routed through a local **Mosquitto MQTT Broker**, acting as the central nervous system for the match.
* **Human Input:** Commands from the Web Controller were transmitted via WebSockets to the broker, then converted to UDP packets for the robots.
* **AI Commands:** The AI Server published velocity vectors for the autonomous robots directly to the broker.

### 3. The Robot Endpoint (ESP32)
Each robot listened to a specific MQTT topic based on its ID. Upon receiving a velocity vector (X, Y), the onboard ESP32 performed the trigonometric mixing required to drive the three omni-wheels, translating abstract vector commands into physical movement instantly.

---

## Technical Implementation

### Web Controller
The human interface was a responsive web application designed to turn any smartphone or laptop into a game controller without requiring app installation.
* **Protocol:** Used the Paho MQTT JavaScript client over WebSockets (Port 9001).
* **Interface:** Featured dual virtual joysticks—Left for translation (movement) and Right for rotation.
* **Optimization:** Implemented input throttling to prevent network flooding while maintaining responsive, low-latency gameplay.

### Robot Firmware
The robots ran custom C++ firmware on **ESP32** microcontrollers.
* **Holonomic Drive:** Implemented Kiwi Drive kinematics to calculate motor speeds for three motors spaced at 120° offsets. This allowed the robots to move in any direction independently of their rotation.
* **Failsafes:** Utilized MQTT "Last Will & Testament" (LWT) protocols to automatically halt robots if they disconnected from the network, preventing runaway hardware during matches.

## Hardware Description
The robots were custom-built omni-directional platforms designed for agility on a small-scale pitch.
* **Chassis:** Custom 3D-printed circular frame.
* **Drive System:** 3x Gear Motors driving custom printed omni-wheels.
* **Electronics:** ESP32 Development Board utilizing L298N motor drivers.
* **Power:** 10x 1.2V NiMH batteries stepped down via the L298N on-board regulators.

---
