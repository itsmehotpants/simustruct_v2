# ⚙️ SimuStruct AI: Real-Time Structural Analysis Surrogate

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![FEniCSx](https://img.shields.io/badge/FEniCSx-FEM%20Solver-lightgrey.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)

This project explores the intersection of **Physics-Informed Engineering** and **Artificial Intelligence**. It replaces a computationally expensive Finite Element Method (FEM) solver with a trained Deep Learning Neural Network (Surrogate Model) to predict stress concentrations in real-time.



## 🚀 The Core Concept
Traditional structural analysis requires solving complex Partial Differential Equations (PDEs) over a meshed grid, which is highly accurate but computationally slow. 

This application bridges that gap by using AI:
1. **Automated Data Generation:** Uses `Gmsh` and `FEniCSx` to simulate 2D plates with randomized elliptical holes under tension, calculating the ground-truth Von Mises stress.
2. **Surrogate Modeling:** Trains a PyTorch Multi-Layer Perceptron (MLP) to learn the physical relationship between the plate's geometry and the resulting stress field.
3. **Real-Time Inference:** A `Streamlit` web dashboard allows users to adjust geometric parameters on the fly, demonstrating how the AI predicts the stress field magnitudes faster than the traditional solver.

## 🛠️ Tech Stack
* **Physics & Meshing:** `FEniCSx` (DOLFINx), `Gmsh`, `meshio`, `UFL`
* **Machine Learning:** `PyTorch`
* **Web Frontend & Visualization:** `Streamlit`, `PyVista`

## 🧠 Neural Network Architecture
The AI is a fully connected Multi-Layer Perceptron (MLP).
* **Input Layer (6 Features):** Node Coordinates (X, Y), Hole Radii (Rx, Ry), and Hole Center (Cx, Cy).
* **Hidden Layers:** 3 Layers (64 -> 128 -> 64 neurons) using ReLU activation.
* **Output Layer (1 Feature):** Predicted Von Mises Stress (in MPa).



## 💻 Installation & Setup

Because physics solvers like FEniCSx rely on complex C++ and MPI dependencies, this project uses a strict Conda environment to prevent package conflicts.

```bash
# 1. Clone the repository
git clone [https://github.com/itsmehotpants/simustruct_v2.git](https://github.com/itsmehotpants/simustruct_v2.git)
cd simustruct_v2

# 2. Create the Conda environment from the provided YAML
conda env create -f environment.yml

# 3. Activate the environment
conda activate simu_ai

## 🎮 How to Use

# 1. Generate the Physics Dataset
Run the automated FEniCSx pipeline to generate `.npz` files containing nodal coordinates, geometric parameters, and ground-truth stress.

python 1_generate_data.py

# 2. Train the AI Brain
Feed the generated dataset into the PyTorch model. This will output a trained stress_model.pth weights file.

python 2_train_ai.py

# 3. Launch the Web Dashboard
Spin up the interactive Streamlit UI to compare the FEM solver against the AI prediction in real-time.

streamlit run app.py
