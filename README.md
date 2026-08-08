# ***Quantum Assisted Physics Informed Neural Network for Nonlinear Fluid Dynamics***

## Selected Challenge

**Computational Physics / Scientific Computing Challenge**: Investigating practical applications of hybrid quantum classical machine learning for solving nonlinear partial differential equations, with a focus on improving physics informed learning workflows for fluid dynamics.

## The Problem

Nonlinear fluid dynamics equations such as the **viscous Burgers’ equation** are widely used as simplified models for shock formation, transport, diffusion, and other complex physical phenomena. Classical numerical solvers require carefully selected spatial and temporal discretization parameters to maintain numerical stability, while conventional physics informed neural networks can become computationally expensive during training because they require repeated automatic differentiation and physics residual evaluation.

The challenge is to develop a computational framework that combines physics based learning with quantum computational components while maintaining a classical numerical solver as a reliable reference.

## Proposed Solution

This project introduces a **Quantum Assisted Physics Informed Neural Network (QAPINN)** framework for approximating solutions to the one dimensional viscous Burgers’ equation.

The architecture combines a classical optimization and neural network workflow with a parameterized quantum circuit. Spatial and temporal coordinates are encoded into quantum rotation angles, and the resulting quantum expectation values are transformed into the predicted physical solution.

A stable **finite difference reference solver** is also implemented to generate a numerical benchmark. The QAPINN predictions are then compared against this reference solution using quantitative error metrics.

## Methods & Tools Considered

* **Quantum Computing Framework**: PennyLane with a simulated parameterized quantum circuit using the `default.qubit` simulator.
* **Quantum Model**: Variational quantum circuit using rotation gates, entangling CNOT operations, and Pauli Z expectation measurements.
* **Machine Learning Framework**: PyTorch for neural network construction, automatic differentiation, optimization, and model training.
* **Classical Numerical Solver**: Finite difference discretization with stability controlled through spatial and temporal step sizes.
* **Physics Informed Learning**: The Burgers’ equation residual is incorporated into the training objective together with initial and boundary condition losses.
* **Evaluation**: Relative L2 error, maximum absolute error, mean absolute error, and prediction comparison against the reference solution.
* **Visualization**: Matplotlib for solution surfaces, error maps, convergence curves, and classical versus quantum assisted predictions.

## Reasoning for the Approach

A hybrid architecture is appropriate because the project does not attempt to replace the entire classical scientific computing pipeline with quantum hardware.

The classical components handle data preparation, numerical reference generation, optimization, and evaluation, while the quantum circuit provides a trainable feature transformation for the physics informed model.

This structure also makes the project practical on current quantum simulators and NISQ oriented workflows. The reference finite difference solution provides an independent numerical baseline, allowing the quantum assisted model to be evaluated using measurable scientific accuracy rather than only training loss.

## Project Workflow

The complete computational pipeline follows these stages:

1. Define the viscous Burgers’ equation and physical parameters.
2. Generate spatial and temporal collocation points.
3. Construct initial and boundary condition datasets.
4. Calculate a stable finite difference reference solution.
5. Train a classical PINN baseline.
6. Construct the parameterized quantum circuit.
7. Encode spatial and temporal inputs into quantum states.
8. Extract quantum expectation values as trainable features.
9. Build and train the QAPINN model.
10. Calculate PDE residual, initial condition, and boundary condition losses.
11. Evaluate both models against the reference solution.
12. Compare accuracy, convergence, and computational behavior.

## Mathematical Model

The project solves the one dimensional viscous Burgers’ equation:

[
u_t + u u_x = \nu u_{xx}
]

where (u(x,t)) represents the physical field and (\nu) represents the viscosity coefficient.

The initial condition is defined as:

[
u(x,0) = -\sin(\pi x)
]

with zero boundary conditions:

[
u(-1,t)=0
]

[
u(1,t)=0
]

The QAPINN is trained by minimizing a combined objective consisting of the PDE residual, initial condition error, and boundary condition error.

## Limitations & Continued Development

* **Quantum Simulation Boundaries**: The current implementation uses a classical simulator for the quantum circuit, so scalability is limited by classical simulation cost.
* **Training Cost**: Evaluating quantum circuits repeatedly during automatic differentiation can make training slower than a purely classical neural network.
* **Limited Qubit Scale**: The experimental model uses a small number of qubits and variational layers rather than a large scale quantum processor.
* **No Demonstrated Quantum Advantage**: The project evaluates the feasibility of the hybrid architecture and does not claim computational quantum advantage.
* **Future Work**: Future development could investigate execution on real quantum hardware, larger quantum circuits, adaptive ansätze, improved sampling strategies, and extensions to higher dimensional fluid dynamics problems.

## Roles & Contributions

* **Santosh (Solo Participant)**: Handled the complete project workflow including problem formulation, numerical reference solver development, classical PINN implementation, quantum circuit design, QAPINN architecture, model training, evaluation, visualization, and comparative analysis.

This format is much closer to your example: **professional, concise, industry/project oriented, and suitable for a GitHub `README.md`**, instead of looking like a full academic thesis.
