# Decision Support System for Autonomous Vehicles Based on Vision-Language Models and Reinforcement Learning

## Overview

This project investigates the integration of Vision-Language Models (VLMs) with Reinforcement Learning (RL) to improve decision-making in autonomous driving scenarios.

The system is developed and evaluated in the CARLA simulator and focuses on challenging driving situations such as:

- Pedestrian crossings
- Collision-risk scenarios
- Rain conditions
- Night driving
- Combined edge cases

The objective is to compare a traditional RL agent against a VLM-assisted RL agent and evaluate whether semantic scene understanding improves driving performance and safety.

---

## Architecture

The project consists of five main modules:

### CARLA Integration

Interface between the simulator and the learning pipeline.

### CARLA Scenarios

Custom driving scenarios used for training and evaluation:

- Pedestrian Collision
- Vehicle Collision Risk
- Rain Mode
- Night Mode
- Combined Edge Cases

### Reinforcement Learning

DQN-based agent responsible for learning driving policies through interaction with the environment.

### Vision-Language Model

Semantic scene understanding module that extracts high-level information from camera observations.

### Reasoning Module

Transforms semantic outputs into structured signals used by the RL agent.

---

## Repository Structure

```text
src/
├── Carla_integration/
├── carla_scenarios/
├── reasoning/
├── rl/
└── vlm/
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd decision-support-autonomous-vehicles
```

Create a virtual environment:

```bash
python3 -m venv carla_env_37
source carla_env_37/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Starting CARLA

Launch CARLA:

```bash
cd CARLA
./CarlaUE4.sh -carla-rpc-port=2000
```

For headless execution:

```bash
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000
```

Verify connection:

```bash
python test_carla.py
```

---

## Training

### RL-only Agent

Example:

```bash
python3 -m src.rl.online.train_rl_online
```

### VLM + RL Agent

Example:

```bash
python3 -m src.rl.online.train_vlm_rl
```

---

## Evaluation

Evaluate the trained agents:

```bash
python3 -m src.rl.online.evaluation_runner
```

Run in background:

```bash
nohup python3 -m src.rl.online.evaluation_runner \
> outputs/logs/evaluation.log 2>&1 &
```

Monitor progress:

```bash
tail -f outputs/logs/evaluation.log
```

---

## Visualization

### Reward Comparison

```bash
python3 src/rl/online/plot_rewards_comparison.py
```

### Cumulative Reward Comparison

```bash
python3 src/rl/online/plot_cumulative_rewards_comparison.py
```

### Survival Steps Comparison

```bash
python3 src/rl/online/plot_survival_steps_comparison.py
```

### Lane Invasions Comparison

```bash
python3 src/rl/online/plot_lane_invasions_comparison.py
```

### Mean Metrics Summary

```bash
python3 src/rl/online/plot_mean_metrics_summary.py
```

---

## Evaluation Metrics

The following metrics are used for comparison:

- Episode Reward
- Cumulative Reward
- Survival Steps
- Number of Collisions
- Number of Lane Invasions
- Average Performance Metrics

---

## Results

The study compares:

- RL-only Agent
- VLM-assisted RL Agent

with a focus on:

- Safety
- Robustness
- Survival Time
- Lane Keeping
- Collision Avoidance

---

## Authors

**Tasnim Selmi**

**Leith Mabrouk**

---

## Supervisor

**Ms. Sameh Najeh**

---

## Academic Year

2025–2026# Decision Support Autonomous Vehicles

