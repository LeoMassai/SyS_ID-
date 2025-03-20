import torch
import matplotlib.pyplot as plt


# Provided TankSystem class
class TankSystem(torch.nn.Module):
    def __init__(self, xbar: torch.Tensor, x_init=None, a: float = 0.6):
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
        #x = x.view(-1, 1, self.state_dim)
        #u = u.view(-1, 1, self.in_dim)
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
            x, _ = self.forward(x, u[:, t:t + 1, :], w[:, t:t + 1, :])  # Use clipped state
            y_traj.append(x)
        y_out = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, state_dim)
        return y_out


# Set parameters
xbar = torch.tensor([1.0])  # Dummy value, not used in dynamics but required by __init__
tank_system = TankSystem(xbar=xbar)
horizon = 200
num_train = 400
num_val = 200
std_noise = 0.002


# Function to generate piecewise constant inputs
def generate_piecewise_constant_inputs(num_traj, horizon, num_segments=5, min_val=0.0, max_val=2.0):
    """Generate piecewise constant inputs with specified number of segments."""
    seg_len = horizon // num_segments  # 200 / 5 = 40 steps per segment
    u_segments = torch.rand(num_traj, num_segments, 1) * (max_val - min_val) + min_val
    u = torch.repeat_interleave(u_segments, seg_len, dim=1)  # Shape: (num_traj, horizon, 1)
    return u


# Function to generate sinusoidal inputs with random phase
def generate_sinusoidal_inputs(num_traj, horizon, omega=2 * torch.pi / 40, amplitude=1.0):
    """Generate sinusoidal inputs with approximately 5 cycles over horizon."""
    t = torch.arange(horizon).float()  # Time steps: 0 to 199
    phase = torch.rand(num_traj, 1, 1) * 2 * torch.pi  # Random phase per trajectory
    u = amplitude * (1 + torch.sin(omega * t[None, :, None] + phase))  # Shape: (num_traj, horizon, 1)
    return u


# Generate training inputs: half piecewise, half sinusoidal
u_train_piecewise = generate_piecewise_constant_inputs(num_train // 2, horizon)
u_train_sinusoidal = generate_sinusoidal_inputs(num_train // 2, horizon)
u_train = torch.cat([u_train_piecewise, u_train_sinusoidal], dim=0)  # Shape: (400, 200, 1)

# Generate validation inputs: half piecewise, half sinusoidal
u_val_piecewise = generate_piecewise_constant_inputs(num_val // 2, horizon)
u_val_sinusoidal = generate_sinusoidal_inputs(num_val // 2, horizon)
u_val = torch.cat([u_val_piecewise, u_val_sinusoidal], dim=0)  # Shape: (200, 200, 1)

# Generate additive white Gaussian noise
w_train = torch.normal(0, std_noise, size=(num_train, horizon, 1))
w_val = torch.normal(0, std_noise, size=(num_val, horizon, 1))

# Simulate trajectories
x_train = tank_system.simulate(u_train, w_train)  # Shape: (400, 200, 1)
x_val = tank_system.simulate(u_val, w_val)  # Shape: (200, 200, 1)

# Save trajectories
torch.save({'u': u_train, 'x': x_train}, 'train_trajectories.pt')
torch.save({'u': u_val, 'x': x_val}, 'val_trajectories.pt')

# Plot sample trajectories
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sample_indices = [0, 1, num_train // 2, num_train // 2 + 1]  # Two piecewise, two sinusoidal
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


# Generate constant input for one trajectory
u_constant = torch.ones((1, horizon, 1)) * 2.0  # Constant input of 2.0

# Generate noise for this trajectory
w_constant = torch.normal(0, std_noise, size=(1, horizon, 1))

# Simulate the trajectory
x_constant = tank_system.simulate(u_constant, w_constant)

# Plot the constant input trajectory
plt.figure(figsize=(6, 4))
time = torch.arange(horizon)
plt.plot(time, x_constant[0, :, 0], label='State $x$ (constant u=2.0)')
plt.plot(time, u_constant[0, :, 0], label='Input $u$', linestyle='--')
plt.title('Trajectory with Constant Input')
plt.xlabel('Time step k')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()




Ts
