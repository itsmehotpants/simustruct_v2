# ⚙️ SimuStruct AI — Real-Time Structural Analysis Surrogate

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FEniCSx-FEM%20Solver-lightgrey.svg" alt="FEniCSx"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg" alt="Streamlit"/>
</p>

> **Replacing a computationally expensive FEM solver with a trained Deep Learning surrogate model to predict stress concentrations in real-time.**

This project sits at the intersection of **Physics-Informed Engineering** and **Artificial Intelligence** — using a Neural Network to approximate the Von Mises stress field in 2D plates, dramatically cutting computation time without sacrificing predictive accuracy.

---

## 📌 Table of Contents

- [Core Concept](#-the-core-concept)
- [Tech Stack](#️-tech-stack)
- [Neural Network Architecture](#-neural-network-architecture)
- [Installation & Setup](#-installation--setup)
- [How to Use](#-how-to-use)
- [Project Structure](#-project-structure)

---

## 🚀 The Core Concept

Traditional structural analysis solves complex **Partial Differential Equations (PDEs)** over a meshed grid — highly accurate, but computationally slow.

SimuStruct AI bridges that gap by learning the physics directly:

| Step | Description |
|------|-------------|
| **1. Data Generation** | Uses `Gmsh` + `FEniCSx` to simulate 2D plates with randomized elliptical holes under tension, computing ground-truth Von Mises stress fields. |
| **2. Surrogate Modeling** | Trains a PyTorch MLP to learn the physical relationship between plate geometry and the resulting stress field. |
| **3. Real-Time Inference** | A `Streamlit` dashboard lets users tweak geometric parameters live, with the AI returning predictions orders of magnitude faster than the FEM solver. |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Physics & Meshing** | `FEniCSx` (DOLFINx), `Gmsh`, `meshio`, `UFL` |
| **Machine Learning** | `PyTorch` |
| **Visualization & Frontend** | `Streamlit`, `PyVista` |

---

## 🧠 Neural Network Architecture

The surrogate is a fully connected **Multi-Layer Perceptron (MLP)**:

```
Input (6)  →  [64]  →  [128]  →  [64]  →  Output (1)
             ReLU      ReLU      ReLU
```

| Layer | Details |
|-------|---------|
| **Input (6 features)** | Node coordinates `(X, Y)`, hole radii `(Rx, Ry)`, hole center `(Cx, Cy)` |
| **Hidden layers** | 3 layers: 64 → 128 → 64 neurons, ReLU activation |
| **Output (1 feature)** | Predicted Von Mises stress (MPa) |

---

## 💻 Installation & Setup

> ⚠️ **Note:** `FEniCSx` relies on complex C++ and MPI dependencies. A **Conda environment** is strongly recommended to avoid package conflicts.

```bash
# 1. Clone the repository
git clone https://github.com/itsmehotpants/simustruct_v2.git
cd simustruct_v2

# 2. Create the Conda environment from the provided YAML
conda env create -f environment.yml

# 3. Activate the environment
conda activate simu_ai
```

---

## 🎮 How to Use

Run the pipeline in three sequential steps:

### Step 1 — Generate the Physics Dataset

Runs the automated FEniCSx simulation pipeline. Outputs `.npz` files containing nodal coordinates, geometric parameters, and ground-truth stress values.

```bash
python 1_generate_data.py
```

### Step 2 — Train the Surrogate Model

Feeds the generated dataset into the PyTorch model. Outputs a trained `stress_model.pth` weights file.

```bash
python 2_train_ai.py
```

### Step 3 — Launch the Web Dashboard

Spins up the interactive Streamlit UI to compare the FEM solver against AI predictions in real-time.

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
simustruct_v2/
├── 1_generate_data.py      # FEniCSx simulation pipeline
├── 2_train_ai.py           # PyTorch model training script
├── app.py                  # Streamlit web dashboard
├── environment.yml         # Conda environment specification
├── stress_model.pth        # Trained model weights (generated)
└── data/                   # Simulation output files (.npz)
```

---

<p align="center">
  Built with physics, powered by AI.
</p>
