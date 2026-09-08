# SIH26054 — Problem Statement (reference copy)

**Organization:** DRDO
**Theme:** Robotics & Drones
**Title:** AI-Enabled Real-Time Digital Twin System for Health Monitoring,
Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines
used in MALE UAVs
**Deadline:** 20 Sept 2026

Kept here verbatim so the whole team works from the same source instead of
paraphrased summaries. Original text supplied by the team lead.

## Background

Medium Altitude Long Endurance (MALE) UAVs are increasingly deployed for
long-duration ISR, communication relay, maritime surveillance, and strategic
defence missions. Reliability and availability of propulsion systems are
critical for mission success, because piston-engine failures during flight
may lead to mission abort, asset loss, or unsafe recovery conditions.

Conventional engine monitoring systems used in UAVs are primarily
threshold-based and reactive — they indicate failures only after an
abnormality has already occurred, with limited capability to estimate
Remaining Useful Life (RUL), predict degradation trends, or simulate
mission-wise engine behavior under varying environmental and operating
conditions.

A Digital Twin (DT) framework for aero piston engines can significantly
improve predictive maintenance, operational reliability, mission planning,
and life cycle management by creating a continuously synchronized virtual
representation of the physical engine using real-time sensor data,
physics-based models, and AI/ML techniques.

The proposed problem aims to develop an indigenous Digital Twin framework
suitable for deployment in MALE UAV ground control and health monitoring
architecture, supporting real-time engine state estimation, anomaly
detection, degradation tracking, fault prediction, and mission replay
capability.

## Description

Develop a scalable and modular Digital Twin system for an aero piston
engine used in MALE UAV applications, integrating: engine sensor data,
thermodynamic behavior models, engine performance maps, failure/degradation
logic, and AI/ML-based predictive analytics.

The system should be capable of:
- Real-time engine parameter visualization
- Monitoring of engine health indicators
- Detection of abnormal operating conditions
- Predicting probable failures before occurrence
- Estimating degradation trends and RUL
- Simulating engine behavior under different mission profiles and
  environmental conditions
- Supporting post-flight analysis and mission replay

The system may utilize CAN bus/SocketCAN-based data acquisition, ECU/FADEC
communication interfaces, edge computing, cloud or local server analytics,
AI/ML anomaly detection, physics-informed modelling, and a dashboard/HMI for
operators and maintenance engineers.

## Expected Solution

The Digital Twin core framework acts as the central intelligence layer that
continuously mirrors the real aero-piston engine operating onboard the MALE
UAV, using live telemetry, physics-based models, operational history, and
AI-driven analytics — designed with future deployment in defence-grade
Ground Control Stations, engine test rigs, and fleet-level health monitoring
infrastructure in mind.

### A. Digital Twin Core Framework
- Virtual engine model synchronized with live engine data
- Modular architecture for future scalability
- Real-time data ingestion capability

### B. Health Monitoring System
Continuously assesses the condition of engine sub-systems and generates
health indices. Monitored parameters:
- RPM
- Cylinder Head Temperature (CHT)
- Exhaust Gas Temperature (EGT)
- Oil pressure & temperature
- Fuel flow
- Vibration signatures
- Battery/alternator health
- Injection timing parameters

### C. Fault Detection & Predictive Analytics
Transition from threshold-based monitoring to intelligent predictive
diagnostics, covering:
- Misfire conditions
- Injector abnormalities
- Coating/coding degradation
- Lubrication issues
- Sensor drift/failure
- Combustion instability
- Overheating trends
- Abnormal vibration patterns

### D. AI/ML Layer
Adaptive learning for predictive diagnostics and maintenance planning:
- Anomaly detection algorithms
- Remaining Useful Life (RUL) estimation
- Trend analysis
- Predictive maintenance recommendations

### E. Simulation & Replay Capability
- Replay of historical mission data
- Environmental condition simulation
- Engine behavior simulation during high-altitude, endurance mission,
  hot-weather operation, and rapid throttle transitions

### F. Visualization Dashboard
Operational interface for UAV operators, propulsion engineers, and
maintenance teams, supporting:
- Real-time engine health status
- Fault alerts
- Engine efficiency trends
- Maintenance advisory
- Mission-wise health reports

## Deliverables

- Functional prototype/software demonstrator
- Digital twin architecture design
- Engine simulation model
- AI/ML-based anomaly detection module
- Visualization dashboard
- Demonstration using simulated or real engine datasets
- Technical documentation and deployment roadmap

## Desired innovation areas

Physics-informed AI, Edge AI for UAV applications, lightweight onboard
analytics, hybrid thermodynamic + data-driven models, federated learning,
explainable AI for fault diagnosis, secure telemetry architecture,
autonomous maintenance advisory systems.

## Technical expectations from participants

IC engine fundamentals, UAV propulsion systems, sensor fusion, embedded
systems, CAN communication, AI/ML analytics, data visualization, simulation
modelling, reliability engineering.
