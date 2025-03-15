# CSCI B659: Reinforcement Learning - Assignment 1
**Author:** LJ Huang  
**Semester:** Spring 2025

## Overview
This repository contains the code and report for Assignment 1. The assignment covers three main tasks:
1. **Task 1:** Implementation and evaluation of Value Iteration (VI), Policy Iteration (PI), and Modified Policy Iteration (MPI) on the Frozen Lake environment.
2. **Task 2:** Exploration of model-based RL by generating corrupted versions of the transition model, planning using VI, and evaluating the resulting policies.
3. **Task 3:** Implementation of a naive model-based RL approach by collecting data with random actions, estimating the environment model, and computing a policy using VI.

## Directory Structure
- **VI_PI_MPI.py**  
  Contains the implementation and experimental driver code for Task 1 (VI, PI, and MPI).
- **corrupt_version.py**  
  Contains the code for Task 2, including the procedure to corrupt the transition model and the associated experiments.
- **Naive_MBRL.py**  
  Contains the implementation for Task 3 (naive model-based RL), including data collection, model estimation, planning, and evaluation.
- **pp1starter.py**  
  The provided startup file for setting up the Frozen Lake environment (including reward and transition models, number of states, and actions).
- **hw1_report.pdf**  
  The report containing the code printouts, experimental results, plots, and discussion of the findings.
- **README.md**  
  This file.

## Dependencies
- Python 3.x
- Gymnasium (Install with `pip install gymnasium`)
- NumPy (Install with `pip install numpy`)
- Matplotlib (Install with `pip install matplotlib`)

## Installation and Running
1. **Clone or Download the Repository:**
   Clone the repository or download the zip file and extract its contents.

2. **Run the Code**
   python VI_PI_MPI.py (Task 1)
   python corrupt_version.py (Task 2)
   python Naive_MBRL.py (Task 3)
