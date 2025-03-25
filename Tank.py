import matplotlib.pyplot as plt
import torch


# Provided TankSystem class
class TankSystem(torch.nn.Module):
    def __init__(self, x_init=None, a: float = 0.6):
        super().__init__()
        self.x_init = torch.tensor(1.1) if x_init is None else x_init.reshape(1, -1)
        self.state_dim = 1
        self.in_dim = 1
        self.h = 0.1
        self.a = a
        self.b = 0.5

    def dynamics(self, x, u):
        f = 1 / (self.b + x) * (-self.a * torch.sqrt(x) + self.a * torch.sqrt(u))
        return f

    def noiseless_forward(self, x: torch.Tensor, u: torch.Tensor):
        f = self.dynamics(x, u)
        x_ = x + self.h * f
        return x_

    def forward(self, x, u, w):
        x_plus = self.noiseless_forward(x, u) + w.view(-1, 1, self.state_dim)
        return torch.relu(x_plus), x_plus

    def simulate(self, u, w):
        horizon = u.shape[1]
        batch_size = u.shape[0]
        y_traj = []
        x = self.x_init  # Shape: (1, state_dim), will broadcast to batch_size
        for t in range(horizon):
            x, _ = self.forward(x, u[:, t:t + 1, :], w[:, t:t + 1, :])
            y_traj.append(x)
        y_out = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, state_dim)
        return y_out


# Set parameters
tank_system = TankSystem()
horizon = 200
num_train = 400
num_val = 200
std_noise = 0.003


# Function to generate piecewise constant inputs
def generate_piecewise_constant_inputs(num_traj, horizon, num_segments=5, min_val=0.0, max_val=2.0):
    seg_len = horizon // num_segments
    u_segments = torch.rand(num_traj, num_segments, 1) * (max_val - min_val) + min_val
    u = torch.repeat_interleave(u_segments, seg_len, dim=1)
    return u


# Function to generate sinusoidal inputs with random phase
def generate_sinusoidal_inputs(num_traj, horizon, omega=2 * torch.pi / 40, amplitude=1.0):
    t = torch.arange(horizon).float()
    phase = torch.rand(num_traj, 1, 1) * 2 * torch.pi
    u = amplitude * (1 + torch.sin(omega * t[None, :, None] + phase))
    return u


# New function: Generate exotic inputs for validation trajectories
def generate_exotic_inputs(num_traj, horizon):
    """
    Generate exotic inputs that differ from training inputs.
    Each trajectory randomly picks one of three types:
      - Chirp signal (frequency increases linearly over time)
      - Sum of two sinusoids with different frequencies
      - Modulated square wave
    The generated signal is scaled to the range [0, 2].
    """
    t = torch.linspace(0, 1, horizon)
    inputs = torch.zeros(num_traj, horizon, 1)
    for i in range(num_traj):
        choice = torch.randint(0, 3, (1,)).item()
        if choice == 0:
            # Chirp signal: frequency increases linearly
            freq_start = 1.0
            freq_end = 10.0
            # Create a frequency that linearly increases
            freqs = freq_start + (freq_end - freq_start) * t
            phase = torch.rand(1) * 2 * torch.pi
            signal = torch.sin(2 * torch.pi * freqs * t + phase)
        elif choice == 1:
            # Sum of two sinusoids with different frequencies and phases
            phase1 = torch.rand(1) * 2 * torch.pi
            phase2 = torch.rand(1) * 2 * torch.pi
            signal = 0.5 * torch.sin(2 * torch.pi * 3 * t + phase1) + 0.5 * torch.sin(2 * torch.pi * 7 * t + phase2)
        else:
            # Modulated square wave: square wave modulated by a slow sine
            square_wave = torch.sign(torch.sin(2 * torch.pi * 5 * t))
            modulator = 0.5 * torch.sin(2 * torch.pi * 0.5 * t) + 0.5  # Varies between 0 and 1
            signal = square_wave * modulator

        # Scale signal to be in the range [0, 2]
        signal_min = signal.min()
        signal_max = signal.max()
        signal = (signal - signal_min) / (signal_max - signal_min + 1e-6) * 2.0
        inputs[i, :, 0] = signal
    return inputs


# Generate training inputs: half piecewise, half sinusoidal
u_train_piecewise = generate_piecewise_constant_inputs(num_train // 2, horizon)
u_train_sinusoidal = generate_sinusoidal_inputs(num_train // 2, horizon)
u_train = torch.cat([u_train_piecewise, u_train_sinusoidal], dim=0)

# Generate validation inputs using the exotic input generator
u_val = generate_exotic_inputs(num_val, horizon)

# Generate additive white Gaussian noise
w_train = torch.normal(0, std_noise, size=(num_train, horizon, 1))
w_val = torch.normal(0, std_noise, size=(num_val, horizon, 1))

# Simulate trajectories
x_train = tank_system.simulate(u_train, w_train)
x_val = tank_system.simulate(u_val, w_val)

# Save trajectories
torch.save({'u': u_train, 'x': x_train}, 'train_trajectories.pt')
torch.save({'u': u_val, 'x': x_val}, 'val_trajectories.pt')

# Plot sample trajectories from training data
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sample_indices = [0, 1, num_train // 2, num_train // 2 + 1]
time = torch.arange(horizon)

for i, idx in enumerate(sample_indices):
    ax = axes[i // 2, i % 2]
    ax.plot(time, x_train[idx, :, 0], label='State $x$')
    ax.plot(time, u_train[idx, :, 0], label='Input $u$', linestyle='--')
    ax.set_title(f'Trajectory {idx + 1}')
    ax.set_xlabel('Time step k')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

print("Trajectories generated and saved as 'train_trajectories.pt' and 'val_trajectories.pt'")

# Additional plotting for constant and sinusoidal inputs (for reference)
horizon_plot = 800
time = torch.arange(horizon_plot)

u_sin = generate_sinusoidal_inputs(1, horizon_plot)
u_constant = torch.ones((1, horizon_plot, 1)) * 2.0
w_constant = torch.normal(0, std_noise, size=(1, horizon_plot, 1))

x_constant = tank_system.simulate(u_constant, w_constant)
x_sin = tank_system.simulate(u_sin, w_constant)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(time, x_constant[0, :, 0], label='State $x$ (constant u=2.0)')
axes[0].plot(time, u_constant[0, :, 0], label='Input $u$', linestyle='--')
axes[0].set_title('Trajectory with Constant Input')
axes[0].set_xlabel('Time step k')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time, x_sin[0, :, 0], label='State $x$ (sinusoidal u)')
axes[1].plot(time, u_sin[0, :, 0], label='Input $u$', linestyle='--')
axes[1].set_title('Trajectory with Sinusoidal Input')
axes[1].set_xlabel('Time step k')
axes[1].set_ylabel('Value')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("trajectories.pdf", format="pdf", bbox_inches="tight")
plt.show()


Ts
