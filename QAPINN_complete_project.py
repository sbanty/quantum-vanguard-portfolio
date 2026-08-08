"""
Quantum Assisted Physics Informed Neural Network (QAPINN)
for the 1D Viscous Burgers Equation

Project workflow
1. Classical PINN baseline
2. Numerical reference solution
3. Variational quantum circuit
4. Quantum feature encoding
5. QAPINN construction
6. QAPINN training
7. Classical vs QAPINN comparison
8. Results and plots

Recommended environment
qBraid notebook with Python 3.10+.

Install missing packages in a qBraid cell if necessary:
    %pip install numpy matplotlib scipy torch pennylane

Equation
    u_t + u u_x = nu u_xx

Domain
    x in [-1, 1]
    t in [0, 1]

Initial condition
    u(x, 0) = -sin(pi x)

Boundary conditions
    u(-1, t) = 0
    u(1, t) = 0
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
# Section 0: Configuration
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# Burgers equation parameters
x_min = -1.0
x_max = 1.0

t_min = 0.0
t_max = 1.0

nu = 0.01 / np.pi

# PINN grid
Nx = 101
Nt = 101

# Classical PINN
classical_hidden = [32, 32, 32]
classical_epochs = 1000
classical_learning_rate = 1e-3

# Training points
N_f = 1000
N_ic = 101
N_bc = 101

# Quantum model
n_qubits = 4
n_layers = 2

# Quantum training
quantum_epochs = 300
quantum_learning_rate = 1e-3
quantum_collocation_points = 150

print("Project configuration loaded.")
print("PyTorch:", torch.__version__)
print("PennyLane:", qml.__version__)


# ============================================================
# Section 1.1: Training Data Generation
# ============================================================

# Interior collocation points
x_f = np.random.uniform(
    x_min,
    x_max,
    N_f
)

t_f = np.random.uniform(
    t_min,
    t_max,
    N_f
)

# Initial condition
x_ic = np.linspace(
    x_min,
    x_max,
    N_ic
)

t_ic = np.zeros_like(x_ic)

u_ic = -np.sin(
    np.pi * x_ic
)

# Boundary condition
t_bc = np.linspace(
    t_min,
    t_max,
    N_bc
)

x_bc_left = np.full_like(
    t_bc,
    x_min
)

x_bc_right = np.full_like(
    t_bc,
    x_max
)

u_bc_left = np.zeros_like(t_bc)
u_bc_right = np.zeros_like(t_bc)

# Torch tensors
x_f_torch = torch.tensor(
    x_f.reshape(-1, 1),
    dtype=torch.float32
)

t_f_torch = torch.tensor(
    t_f.reshape(-1, 1),
    dtype=torch.float32
)

x_ic_torch = torch.tensor(
    x_ic.reshape(-1, 1),
    dtype=torch.float32
)

t_ic_torch = torch.tensor(
    t_ic.reshape(-1, 1),
    dtype=torch.float32
)

u_ic_torch = torch.tensor(
    u_ic.reshape(-1, 1),
    dtype=torch.float32
)

x_bc_left_torch = torch.tensor(
    x_bc_left.reshape(-1, 1),
    dtype=torch.float32
)

x_bc_right_torch = torch.tensor(
    x_bc_right.reshape(-1, 1),
    dtype=torch.float32
)

t_bc_torch = torch.tensor(
    t_bc.reshape(-1, 1),
    dtype=torch.float32
)

u_bc_left_torch = torch.tensor(
    u_bc_left.reshape(-1, 1),
    dtype=torch.float32
)

u_bc_right_torch = torch.tensor(
    u_bc_right.reshape(-1, 1),
    dtype=torch.float32
)

print("Section 1.1 complete.")
print("Collocation points:", N_f)


# ============================================================
# Section 1.2: Classical PINN
# ============================================================

class ClassicalPINN(nn.Module):

    def __init__(self, hidden_layers=None):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [32, 32, 32]

        layers = []
        input_size = 2

        for hidden_size in hidden_layers:
            layers.append(
                nn.Linear(
                    input_size,
                    hidden_size
                )
            )
            layers.append(
                nn.Tanh()
            )
            input_size = hidden_size

        layers.append(
            nn.Linear(
                input_size,
                1
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x, t):

        inputs = torch.cat(
            [x, t],
            dim=1
        )

        return self.network(inputs)


classical_model = ClassicalPINN(
    classical_hidden
)

classical_parameters = sum(
    p.numel()
    for p in classical_model.parameters()
    if p.requires_grad
)

print("Classical PINN parameters:", classical_parameters)


# ============================================================
# Section 1.3: Classical PINN Physics Residual
# ============================================================

def classical_pde_residual(model, x, t):

    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = model(x, t)

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = (
        u_t
        + u * u_x
        - nu * u_xx
    )

    return residual


def classical_loss(model):

    residual = classical_pde_residual(
        model,
        x_f_torch,
        t_f_torch
    )

    loss_pde = torch.mean(
        residual ** 2
    )

    u_ic_prediction = model(
        x_ic_torch,
        t_ic_torch
    )

    loss_ic = torch.mean(
        (
            u_ic_prediction
            - u_ic_torch
        ) ** 2
    )

    u_left_prediction = model(
        x_bc_left_torch,
        t_bc_torch
    )

    u_right_prediction = model(
        x_bc_right_torch,
        t_bc_torch
    )

    loss_bc = (
        torch.mean(
            (
                u_left_prediction
                - u_bc_left_torch
            ) ** 2
        )
        +
        torch.mean(
            (
                u_right_prediction
                - u_bc_right_torch
            ) ** 2
        )
    )

    total_loss = (
        loss_pde
        + loss_ic
        + loss_bc
    )

    return (
        total_loss,
        loss_pde,
        loss_ic,
        loss_bc
    )


# ============================================================
# Section 1.4: Train Classical PINN
# ============================================================

classical_optimizer = torch.optim.Adam(
    classical_model.parameters(),
    lr=classical_learning_rate
)

classical_total_losses = []
classical_pde_losses = []
classical_ic_losses = []
classical_bc_losses = []

print("Training classical PINN...")

classical_start = time.time()

classical_model.train()

for epoch in range(classical_epochs):

    classical_optimizer.zero_grad()

    total_loss, pde_loss, ic_loss, bc_loss = classical_loss(
        classical_model
    )

    total_loss.backward()

    classical_optimizer.step()

    classical_total_losses.append(
        total_loss.detach().item()
    )

    classical_pde_losses.append(
        pde_loss.detach().item()
    )

    classical_ic_losses.append(
        ic_loss.detach().item()
    )

    classical_bc_losses.append(
        bc_loss.detach().item()
    )

    if epoch == 0 or (epoch + 1) % 100 == 0:
        print(
            f"Classical epoch {epoch + 1}/{classical_epochs} | "
            f"Loss {total_loss.item():.6e}"
        )

classical_training_time = (
    time.time() - classical_start
)

print(
    f"Classical PINN training time: "
    f"{classical_training_time:.2f} s"
)


# ============================================================
# Section 1.5: Numerical Reference Solution
# ============================================================

Nx_ref = 401

x_ref = np.linspace(
    x_min,
    x_max,
    Nx_ref
)

dx_ref = x_ref[1] - x_ref[0]

dt_diffusion = (
    0.4 * dx_ref**2 / nu
)

dt_convection = (
    0.4 * dx_ref
)

dt_ref = min(
    dt_diffusion,
    dt_convection
)

Nt_ref = int(
    np.ceil(
        (t_max - t_min) / dt_ref
    )
) + 1

dt_ref = (
    (t_max - t_min)
    / (Nt_ref - 1)
)

t_ref = np.linspace(
    t_min,
    t_max,
    Nt_ref
)

diffusion_cfl = (
    nu * dt_ref / dx_ref**2
)

print(
    "Reference diffusion CFL:",
    diffusion_cfl
)

u_ref = np.zeros(
    (Nt_ref, Nx_ref),
    dtype=np.float64
)

u_ref[0, :] = -np.sin(
    np.pi * x_ref
)

u_ref[:, 0] = 0.0
u_ref[:, -1] = 0.0

for n in range(Nt_ref - 1):

    u_old = u_ref[n, :].copy()

    u_x = np.where(
        u_old[1:-1] >= 0.0,
        (
            u_old[1:-1]
            - u_old[:-2]
        ) / dx_ref,
        (
            u_old[2:]
            - u_old[1:-1]
        ) / dx_ref
    )

    u_xx = (
        u_old[2:]
        - 2.0 * u_old[1:-1]
        + u_old[:-2]
    ) / dx_ref**2

    u_new = u_old.copy()

    u_new[1:-1] = (
        u_old[1:-1]
        - dt_ref
        * u_old[1:-1]
        * u_x
        + dt_ref
        * nu
        * u_xx
    )

    u_new[0] = 0.0
    u_new[-1] = 0.0

    u_ref[n + 1, :] = u_new

if (
    np.isnan(u_ref).any()
    or np.isinf(u_ref).any()
):
    raise RuntimeError(
        "Reference solution contains NaN or infinity."
    )

print(
    "Reference solution calculated:",
    u_ref.shape
)


# ============================================================
# Section 1.6: Evaluation Grid
# ============================================================

x_plot = np.linspace(
    x_min,
    x_max,
    Nx
)

t_plot = np.linspace(
    t_min,
    t_max,
    Nt
)

X, T = np.meshgrid(
    x_plot,
    t_plot
)

X_torch = torch.tensor(
    X.reshape(-1, 1),
    dtype=torch.float32
)

T_torch = torch.tensor(
    T.reshape(-1, 1),
    dtype=torch.float32
)


# ============================================================
# Section 1.7: Classical PINN Prediction and Error
# ============================================================

classical_model.eval()

with torch.no_grad():

    classical_prediction = classical_model(
        X_torch,
        T_torch
    )

classical_prediction = (
    classical_prediction
    .cpu()
    .numpy()
    .reshape(Nt, Nx)
)

reference_interpolator = (
    RegularGridInterpolator(
        (t_ref, x_ref),
        u_ref
    )
)

comparison_points = np.column_stack(
    (
        T.reshape(-1),
        X.reshape(-1)
    )
)

U_reference = reference_interpolator(
    comparison_points
).reshape(
    Nt,
    Nx
)

classical_error = (
    classical_prediction
    - U_reference
)

classical_relative_l2 = (
    np.linalg.norm(classical_error)
    /
    np.linalg.norm(U_reference)
)

classical_max_error = np.max(
    np.abs(classical_error)
)

classical_mean_error = np.mean(
    np.abs(classical_error)
)

print("Classical PINN relative L2 error:")
print(classical_relative_l2)


# ============================================================
# Section 1.8: Classical PINN Plots
# ============================================================

plt.figure(figsize=(9, 6))

plt.contourf(
    X,
    T,
    classical_prediction,
    levels=50
)

plt.colorbar(
    label="u(x,t)"
)

plt.xlabel("x")
plt.ylabel("t")
plt.title(
    "Classical PINN Prediction"
)

plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 6))

plt.contourf(
    X,
    T,
    np.abs(classical_error),
    levels=50
)

plt.colorbar(
    label="Absolute Error"
)

plt.xlabel("x")
plt.ylabel("t")
plt.title(
    "Classical PINN Absolute Error"
)

plt.tight_layout()
plt.show()


# ============================================================
# Section 2.1: Quantum Device
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)

print(
    "Quantum device created with",
    n_qubits,
    "qubits."
)


# ============================================================
# Section 2.2: Variational Quantum Circuit
# ============================================================

@qml.qnode(
    dev,
    interface="torch",
    diff_method="backprop"
)
def quantum_circuit(
    inputs,
    weights
):

    for i in range(n_qubits):

        qml.RY(
            inputs[i],
            wires=i
        )

    for layer in range(n_layers):

        for i in range(n_qubits):

            qml.RY(
                weights[layer, i, 0],
                wires=i
            )

            qml.RZ(
                weights[layer, i, 1],
                wires=i
            )

        for i in range(n_qubits - 1):

            qml.CNOT(
                wires=[
                    i,
                    i + 1
                ]
            )

    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(n_qubits)
    ]


# ============================================================
# Section 2.3: Quantum Assisted Layer
# ============================================================

class QuantumLayer(nn.Module):

    def __init__(
        self,
        n_qubits=4,
        n_layers=2
    ):

        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.weights = nn.Parameter(
            0.01 * torch.randn(
                n_layers,
                n_qubits,
                2,
                dtype=torch.float32
            )
        )

        self.output_layer = nn.Linear(
            n_qubits,
            1
        )

    def forward(
        self,
        x,
        t
    ):

        batch_outputs = []

        for i in range(
            x.shape[0]
        ):

            x_i = x[i, 0].to(
                torch.float32
            )

            t_i = t[i, 0].to(
                torch.float32
            )

            features = torch.stack(
                [
                    torch.pi * x_i,
                    torch.pi * t_i,
                    torch.pi * x_i * t_i,
                    torch.pi * (x_i + t_i)
                ]
            ).to(
                torch.float32
            )

            q_output = quantum_circuit(
                features,
                self.weights
            )

            q_output = torch.stack(
                q_output
            ).to(
                torch.float32
            )

            batch_outputs.append(
                q_output
            )

        quantum_features = torch.stack(
            batch_outputs
        ).to(
            torch.float32
        )

        output = self.output_layer(
            quantum_features
        )

        return output


class QAPINN(nn.Module):

    def __init__(
        self,
        n_qubits=4,
        n_layers=2
    ):

        super().__init__()

        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers
        )

    def forward(
        self,
        x,
        t
    ):

        return self.quantum_layer(
            x,
            t
        )


qapinn_model = QAPINN(
    n_qubits=n_qubits,
    n_layers=n_layers
)

qapinn_parameters = sum(
    p.numel()
    for p in qapinn_model.parameters()
    if p.requires_grad
)

print(
    "QAPINN parameters:",
    qapinn_parameters
)


# ============================================================
# Section 2.4: QAPINN PDE Residual
# ============================================================

def qapinn_pde_residual(
    model,
    x,
    t
):

    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = model(
        x,
        t
    )

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = (
        u_t
        + u * u_x
        - nu * u_xx
    )

    return residual


# ============================================================
# Section 2.5: QAPINN Loss
# ============================================================

def qapinn_loss(
    model,
    x_f,
    t_f
):

    residual = qapinn_pde_residual(
        model,
        x_f,
        t_f
    )

    loss_pde = torch.mean(
        residual ** 2
    )

    u_ic_prediction = model(
        x_ic_torch,
        t_ic_torch
    )

    loss_ic = torch.mean(
        (
            u_ic_prediction
            - u_ic_torch
        ) ** 2
    )

    u_left_prediction = model(
        x_bc_left_torch,
        t_bc_torch
    )

    u_right_prediction = model(
        x_bc_right_torch,
        t_bc_torch
    )

    loss_bc = (
        torch.mean(
            (
                u_left_prediction
                - u_bc_left_torch
            ) ** 2
        )
        +
        torch.mean(
            (
                u_right_prediction
                - u_bc_right_torch
            ) ** 2
        )
    )

    total_loss = (
        loss_pde
        + loss_ic
        + loss_bc
    )

    return (
        total_loss,
        loss_pde,
        loss_ic,
        loss_bc
    )


# ============================================================
# Section 3.1: QAPINN Training Data
# ============================================================

max_points = min(
    quantum_collocation_points,
    x_f_torch.shape[0]
)

indices = torch.randperm(
    x_f_torch.shape[0]
)[:max_points]

x_f_quantum = (
    x_f_torch[indices]
)

t_f_quantum = (
    t_f_torch[indices]
)

print(
    "Quantum collocation points:",
    max_points
)


# ============================================================
# Section 3.2: Train QAPINN
# ============================================================

q_optimizer = torch.optim.Adam(
    qapinn_model.parameters(),
    lr=quantum_learning_rate
)

q_total_losses = []
q_pde_losses = []
q_ic_losses = []
q_bc_losses = []

print("Training QAPINN...")

quantum_start = time.time()

qapinn_model.train()

for epoch in range(
    quantum_epochs
):

    q_optimizer.zero_grad()

    total_loss, pde_loss, ic_loss, bc_loss = qapinn_loss(
        qapinn_model,
        x_f_quantum,
        t_f_quantum
    )

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        qapinn_model.parameters(),
        max_norm=1.0
    )

    q_optimizer.step()

    q_total_losses.append(
        total_loss.detach().item()
    )

    q_pde_losses.append(
        pde_loss.detach().item()
    )

    q_ic_losses.append(
        ic_loss.detach().item()
    )

    q_bc_losses.append(
        bc_loss.detach().item()
    )

    if (
        epoch == 0
        or (epoch + 1) % 25 == 0
    ):

        print(
            f"Quantum epoch {epoch + 1}/{quantum_epochs} | "
            f"Loss {total_loss.item():.6e}"
        )

quantum_training_time = (
    time.time() - quantum_start
)

print(
    f"QAPINN training time: "
    f"{quantum_training_time:.2f} s"
)


# ============================================================
# Section 3.3: QAPINN Convergence
# ============================================================

plt.figure(figsize=(8, 5))

plt.semilogy(
    q_total_losses,
    label="Total Loss"
)

plt.semilogy(
    q_pde_losses,
    label="PDE Loss"
)

plt.semilogy(
    q_ic_losses,
    label="IC Loss"
)

plt.semilogy(
    q_bc_losses,
    label="BC Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "QAPINN Training Convergence"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Section 4.1: QAPINN Prediction
# ============================================================

qapinn_model.eval()

# Evaluating the full grid point-by-point can be expensive.
# This is intentionally explicit so it works with the quantum layer.

q_prediction_values = []

with torch.no_grad():

    for start in range(
        0,
        X_torch.shape[0],
        100
    ):

        end = min(
            start + 100,
            X_torch.shape[0]
        )

        batch_prediction = qapinn_model(
            X_torch[start:end],
            T_torch[start:end]
        )

        q_prediction_values.append(
            batch_prediction.cpu()
        )

q_prediction = torch.cat(
    q_prediction_values,
    dim=0
).numpy().reshape(
    Nt,
    Nx
)

print(
    "QAPINN prediction shape:",
    q_prediction.shape
)


# ============================================================
# Section 4.2: QAPINN Accuracy
# ============================================================

q_error = (
    q_prediction
    - U_reference
)

q_relative_l2 = (
    np.linalg.norm(q_error)
    /
    np.linalg.norm(U_reference)
)

q_max_error = np.max(
    np.abs(q_error)
)

q_mean_error = np.mean(
    np.abs(q_error)
)

print(
    "QAPINN relative L2 error:",
    q_relative_l2
)

print(
    "QAPINN maximum error:",
    q_max_error
)

print(
    "QAPINN mean absolute error:",
    q_mean_error
)


# ============================================================
# Section 4.3: Comparison Table
# ============================================================

print()
print("Model comparison")
print()

print(
    f"{'Metric':35s}"
    f"{'Classical PINN':>20s}"
    f"{'QAPINN':>20s}"
)

print(
    f"{'Trainable parameters':35s}"
    f"{classical_parameters:>20d}"
    f"{qapinn_parameters:>20d}"
)

print(
    f"{'Training time (seconds)':35s}"
    f"{classical_training_time:>20.2f}"
    f"{quantum_training_time:>20.2f}"
)

print(
    f"{'Relative L2 error':35s}"
    f"{classical_relative_l2:>20.6e}"
    f"{q_relative_l2:>20.6e}"
)

print(
    f"{'Maximum absolute error':35s}"
    f"{classical_max_error:>20.6e}"
    f"{q_max_error:>20.6e}"
)

print(
    f"{'Mean absolute error':35s}"
    f"{classical_mean_error:>20.6e}"
    f"{q_mean_error:>20.6e}"
)


# ============================================================
# Section 4.4: Prediction Comparison
# ============================================================

target_time = 0.5

time_index = np.argmin(
    np.abs(
        t_plot - target_time
    )
)

plt.figure(figsize=(8, 5))

plt.plot(
    x_plot,
    U_reference[time_index, :],
    label="Numerical Reference"
)

plt.plot(
    x_plot,
    classical_prediction[time_index, :],
    "--",
    label="Classical PINN"
)

plt.plot(
    x_plot,
    q_prediction[time_index, :],
    ":",
    label="QAPINN"
)

plt.xlabel("x")
plt.ylabel("u(x,t)")

plt.title(
    f"Model Comparison at t = {t_plot[time_index]:.2f}"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Section 4.5: QAPINN Error Map
# ============================================================

plt.figure(figsize=(9, 6))

plt.contourf(
    X,
    T,
    np.abs(q_error),
    levels=50
)

plt.colorbar(
    label="Absolute Error"
)

plt.xlabel("x")
plt.ylabel("t")

plt.title(
    "QAPINN Absolute Error"
)

plt.tight_layout()
plt.show()


# ============================================================
# Section 4.6: Loss Comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.semilogy(
    classical_total_losses,
    label="Classical PINN"
)

plt.semilogy(
    q_total_losses,
    label="QAPINN"
)

plt.xlabel("Epoch")
plt.ylabel("Total Loss")

plt.title(
    "Classical PINN vs QAPINN Training Loss"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Section 5: Quantum Architecture Summary
# ============================================================

print()
print("Quantum architecture")
print()

print("Number of qubits:", n_qubits)
print("Variational layers:", n_layers)
print("Encoding features:", 4)
print("Entangling gate: CNOT")
print("Measurement: Pauli-Z expectation values")
print("Quantum simulator: PennyLane default.qubit")


# ============================================================
# Section 6: Final Results Dictionary
# ============================================================

final_results = {
    "equation": "1D viscous Burgers equation",
    "viscosity": nu,
    "classical_pinn_parameters": classical_parameters,
    "qapinn_parameters": qapinn_parameters,
    "classical_epochs": classical_epochs,
    "qapinn_epochs": quantum_epochs,
    "classical_training_time_seconds": classical_training_time,
    "qapinn_training_time_seconds": quantum_training_time,
    "classical_relative_L2_error": classical_relative_l2,
    "qapinn_relative_L2_error": q_relative_l2,
    "classical_max_absolute_error": classical_max_error,
    "qapinn_max_absolute_error": q_max_error,
    "classical_mean_absolute_error": classical_mean_error,
    "qapinn_mean_absolute_error": q_mean_error,
    "qubits": n_qubits,
    "quantum_layers": n_layers
}

print()
print("Final project results")
print()

for key, value in final_results.items():
    print(f"{key}: {value}")


# ============================================================
# Section 7: Save Numerical Results
# ============================================================

np.savez(
    "qapinn_results.npz",
    x=x_plot,
    t=t_plot,
    reference=U_reference,
    classical_prediction=classical_prediction,
    qapinn_prediction=q_prediction,
    classical_error=classical_error,
    qapinn_error=q_error,
    classical_total_loss=np.array(
        classical_total_losses
    ),
    qapinn_total_loss=np.array(
        q_total_losses
    )
)

print()
print("Saved numerical results to qapinn_results.npz")


# ============================================================
# Section 8: Save Model Weights
# ============================================================

torch.save(
    classical_model.state_dict(),
    "classical_pinn_weights.pt"
)

torch.save(
    qapinn_model.state_dict(),
    "qapinn_weights.pt"
)

print("Saved classical_pinn_weights.pt")
print("Saved qapinn_weights.pt")

print()
print("Project execution completed.")
