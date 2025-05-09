import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm
import copy

num_qubits = 8
device_shots = 8192
batch_size = 64
image_size = 28 
latent_dim = 16
n_epochs = 8
critic_iterations = 5
lambda_gp = 10
lr_critic = 0.0001
lr_generator = 0.0001
lr_rl = 0.0005
gamma = 0.99
memory_capacity = 10000
exploration_fraction = 0.2
min_epsilon = 0.01
n_actions_per_qubit = 5 
n_max_layers = 5
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

dev = qml.device("default.qubit", wires=num_qubits, shots=None) 
GATE_OPTIONS = {
    0: "RX",
    1: "RY",
    2: "RZ",
    3: "Hadamard",
    4: "CNOT"
}

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

class CircuitDesignState:
    def __init__(self, num_qubits, max_layers):
        self.num_qubits = num_qubits
        self.max_layers = max_layers
        self.current_layer = 0
        self.gates = [[] for _ in range(max_layers)]
        self.connectivity_matrix = np.zeros((num_qubits, num_qubits))
        self.parameters = []
        self.qubit_usage = np.zeros(num_qubits)
        
    def to_vector(self):
        features = []
        features.append(self.current_layer / self.max_layers)        
        gate_counts = np.zeros(len(GATE_OPTIONS))
        for layer in self.gates:
            for gate in layer:
                if gate[0] in GATE_OPTIONS:
                    gate_counts[gate[0]] += 1
        features.extend(gate_counts / (self.num_qubits * self.max_layers))
        
        features.extend(self.qubit_usage / self.max_layers)
        connectivity_features = self.connectivity_matrix.flatten()
        features.extend(connectivity_features)
        
        return np.array(features, dtype=np.float32)
    
    def is_complete(self):
        return self.current_layer >= self.max_layers
    
    def add_gate(self, gate_type, target_qubit, control_qubit=None):
        if self.current_layer < self.max_layers:
            if gate_type == 4:  # CNOT
                if control_qubit is not None:
                    self.gates[self.current_layer].append((gate_type, control_qubit, target_qubit))
                    self.connectivity_matrix[control_qubit, target_qubit] += 1
                    self.qubit_usage[control_qubit] += 1
                    self.qubit_usage[target_qubit] += 1
                    self.parameters.append(None) # No parameter for CNOT
            else:
                self.gates[self.current_layer].append((gate_type, target_qubit))
                self.qubit_usage[target_qubit] += 1
                self.parameters.append(None) # Placeholder, will be optimized during training
                
    def next_layer(self):
        if self.current_layer < self.max_layers:
            self.current_layer += 1
            return True
        return False

class CircuitDesignAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CircuitDesignAgent, self).__init__()
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        action_probs = self.policy_network(state)
        state_value = self.value_network(state)
        return action_probs, state_value
    
    def act(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.choice(len(GATE_OPTIONS) * num_qubits * (num_qubits + 1))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.forward(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample().item()
        return action
    
    def evaluate(self, states, actions):
        state_tensor = torch.FloatTensor(states)
        action_probs, state_values = self.forward(state_tensor)
        
        action_dist = Categorical(action_probs)
        action_log_probs = action_dist.log_prob(actions)
        dist_entropy = action_dist.entropy()
        
        return action_log_probs, state_values, dist_entropy

def action_to_gate_and_qubits(action, num_qubits):
    total_options = len(GATE_OPTIONS) * num_qubits * (num_qubits + 1)
    if action >= total_options:
        action = total_options - 1
    gate_type = action % len(GATE_OPTIONS)
    remaining = action // len(GATE_OPTIONS)
    target_qubit = remaining % num_qubits
    remaining = remaining // num_qubits
    control_qubit = remaining % (num_qubits + 1)
    if control_qubit == num_qubits or gate_type != 4: 
        control_qubit = None
    elif control_qubit == target_qubit:
        control_qubit = (control_qubit + 1) % num_qubits
    
    return gate_type, target_qubit, control_qubit

def dynamic_quantum_generator(latent_vector, circuit_design, params):
    latent_vector = np.tanh(latent_vector) * 0.5  # Maps to [-0.5, 0.5]
    for i, value in enumerate(latent_vector):
        if i < num_qubits:
            qml.RY(value * np.pi, wires=i)
    
    param_idx = 0
    for layer_idx, layer in enumerate(circuit_design.gates):
        if not layer: # Skip empty layers
            continue
            
        for gate in layer:
            gate_type = gate[0]
            if param_idx < len(params):
                param_value = np.tanh(params[param_idx].item()) * np.pi  # Use tanh to constrain
            else:
                param_value = 0.0
                
            if gate_type == 0:  # RX
                qml.RX(param_value, wires=gate[1])
                if gate_type in [0, 1, 2]:  # Only parameterized gates
                    param_idx += 1
            elif gate_type == 1:  # RY
                qml.RY(param_value, wires=gate[1])
                if gate_type in [0, 1, 2]:
                    param_idx += 1
            elif gate_type == 2:  # RZ
                qml.RZ(param_value, wires=gate[1])
                if gate_type in [0, 1, 2]:
                    param_idx += 1
            elif gate_type == 3:  # Hadamard
                qml.Hadamard(wires=gate[1])
            elif gate_type == 4 and len(gate) > 2:  # CNOT
                qml.CNOT(wires=[gate[1], gate[2]])
    
    remaining_params = min(num_qubits, len(params) - param_idx)
    for i in range(remaining_params):
        if param_idx + i < len(params):
            param_value = np.tanh(params[param_idx + i].item()) * np.pi
            qml.RX(param_value, wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

class ClassicalCritic(nn.Module):
    def __init__(self, input_size):
        super(ClassicalCritic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, img):
        return self.model(img)

def compute_gradient_penalty(critic, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def compute_inception_score(images, n_split=10, eps=1e-16):
    split_size = images.shape[0] // n_split
    scores = []
    
    for i in range(n_split):
        start = i * split_size
        end = start + split_size
        subset = images[start:end]
        var_across_images = torch.var(subset, dim=0).mean()
        scores.append(var_across_images.item())
    
    return np.mean(scores)

def train_rl_agent_to_design_circuit(max_iterations=1000):
    initial_design = CircuitDesignState(num_qubits, n_max_layers)
    # Initialize the RL agent
    state_dim = (1 + len(GATE_OPTIONS) + num_qubits + num_qubits**2)
    action_dim = len(GATE_OPTIONS) * num_qubits * (num_qubits + 1)
    rl_agent = CircuitDesignAgent(state_dim, action_dim)
    # Set up optimizer for the RL agent
    rl_optimizer = optim.Adam(rl_agent.parameters(), lr=lr_rl)
    # Create replay buffer
    replay_buffer = ReplayBuffer(memory_capacity)
    current_design = copy.deepcopy(initial_design)
    best_design = None
    best_reward = float('-inf')
    
    # Training loop
    for iteration in tqdm(range(max_iterations), desc="Training RL Agent"):
        # Reset the circuit design
        current_design = copy.deepcopy(initial_design)
        episode_rewards = []
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_values = []
        episode_entropies = []
        epsilon = max(min_epsilon, (1.0 - iteration / (exploration_fraction * max_iterations)))
        
        done = False
        while not done:
            # Get current state
            state = current_design.to_vector()
            episode_states.append(state)
            # Select an action
            action = rl_agent.act(state, epsilon)
            episode_actions.append(action)
            # Extract gate information from action
            gate_type, target_qubit, control_qubit = action_to_gate_and_qubits(action, num_qubits)
            # Apply action to the circuit design
            current_design.add_gate(gate_type, target_qubit, control_qubit)
            # Check if we should move to the next layer
            if len(current_design.gates[current_design.current_layer]) >= num_qubits:
                # Move to next layer if we've added enough gates
                next_layer = current_design.next_layer()
                if not next_layer:
                    done = True
            
            if current_design.is_complete():
                done = True
            temp_params = np.random.uniform(-np.pi, np.pi, size=num_qubits * 3)
            reward = 0
            if done:
                total_gates = sum(len(layer) for layer in current_design.gates)
                connectivity = np.sum(current_design.connectivity_matrix) / (num_qubits * (num_qubits - 1))
                param_efficiency = len([p for p in current_design.parameters if p is not None]) / max(1, total_gates)
                qubit_usage_variance = np.var(current_design.qubit_usage)
                
                # Calculate reward components
                gate_reward = -0.01 * total_gates  # Penalize too many gates
                connectivity_reward = 5.0 * connectivity  # Reward good connectivity
                efficiency_reward = 3.0 * param_efficiency  # Reward parameter efficiency
                balancing_reward = -0.5 * qubit_usage_variance  # Reward balanced qubit usage
                
                reward = gate_reward + connectivity_reward + efficiency_reward + balancing_reward                
                if reward > best_reward:
                    best_reward = reward
                    best_design = copy.deepcopy(current_design)
            
            episode_rewards.append(reward)            
            next_state = current_design.to_vector() if not done else None
            replay_buffer.push(state, action, reward, next_state, done)
        
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            action_log_probs, state_values, dist_entropy = rl_agent.evaluate(states, actions_tensor)            
            advantages = rewards_tensor - state_values.squeeze()
            actor_loss = -(action_log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            entropy_loss = -0.01 * dist_entropy.mean()
            total_loss = actor_loss + critic_loss + entropy_loss
            
            rl_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(rl_agent.parameters(), 0.5)
            rl_optimizer.step()
    
    print(f"Best circuit design reward: {best_reward}")
    return best_design

def create_quantum_generator_from_design(circuit_design):
    n_params = sum(1 for layer in circuit_design.gates for gate in layer 
                   if gate[0] in [0, 1, 2])  # RX, RY, RZ
    
    n_params += num_qubits
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_generator(latent_vector, params):
        return dynamic_quantum_generator(latent_vector, circuit_design, params)
    
    return quantum_generator, n_params

def quantum_generator_output_to_image(quantum_output, image_size):
    # Scale the quantum output from [-1, 1] to [0, 1]
    scaled_output = (quantum_output + 1) / 2
    # For MNIST, we can reshape to match the flattened image
    reshaped_output = torch.zeros(image_size * image_size)
    # Map the quantum output to pixels
    for i in range(min(len(quantum_output), len(reshaped_output))):
        reshaped_output[i] = scaled_output[i]
    if len(quantum_output) < len(reshaped_output):
        repeat_factor = len(reshaped_output) // len(quantum_output)
        for i in range(len(quantum_output)):
            for j in range(repeat_factor):
                idx = i * repeat_factor + j
                if idx < len(reshaped_output):
                    reshaped_output[idx] = scaled_output[i]
    
    return reshaped_output

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

class PatchCorrelationCounter:
    def __init__(self, image_size, patch_size=7):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.counter = np.zeros(self.n_patches)
        self.patch_map = np.zeros((image_size, image_size), dtype=int)
        self.boundary_values = {}
        patch_idx = 0
        for i in range(0, image_size, patch_size):
            for j in range(0, image_size, patch_size):
                if i + patch_size <= image_size and j + patch_size <= image_size:
                    self.patch_map[i:i+patch_size, j:j+patch_size] = patch_idx
                    patch_idx += 1
    
    def reset(self):
        self.counter = np.zeros(self.n_patches)
        self.boundary_values = {}
    
    def get_boundary_condition(self, patch_idx):
        if patch_idx in self.boundary_values:
            return self.boundary_values[patch_idx]
        
        counter_value = self.counter[patch_idx]
        patch_row = patch_idx // (self.image_size // self.patch_size)
        patch_col = patch_idx % (self.image_size // self.patch_size)        
        boundary_condition = np.zeros(4)# [top, right, bottom, left]
        boundary_condition[:] = np.tanh(counter_value) * 0.5
        
        return boundary_condition
    
    def update_counter(self, patch_idx, patch_pixels):
        patch_2d = patch_pixels.view(self.patch_size, self.patch_size).detach()
        top_edge = patch_2d[0, :].mean().item()
        right_edge = patch_2d[:, -1].mean().item()
        bottom_edge = patch_2d[-1, :].mean().item()
        left_edge = patch_2d[:, 0].mean().item()
        
        boundary_values = np.array([top_edge, right_edge, bottom_edge, left_edge])
        self.boundary_values[patch_idx] = boundary_values        
        self.counter[patch_idx] = np.mean(boundary_values) * 2.0
        
        patch_row = patch_idx // (self.image_size // self.patch_size)
        patch_col = patch_idx % (self.image_size // self.patch_size)
        patches_per_row = self.image_size // self.patch_size        
        if patch_row > 0:
            top_neighbor = (patch_row - 1) * patches_per_row + patch_col
            self.counter[top_neighbor] = (self.counter[top_neighbor] + top_edge) / 2
            
        # Right neighbor
        if patch_col < patches_per_row - 1:
            right_neighbor = patch_row * patches_per_row + patch_col + 1
            self.counter[right_neighbor] = (self.counter[right_neighbor] + right_edge) / 2
            
        # Bottom neighbor
        if patch_row < patches_per_row - 1:
            bottom_neighbor = (patch_row + 1) * patches_per_row + patch_col
            self.counter[bottom_neighbor] = (self.counter[bottom_neighbor] + bottom_edge) / 2
            
        # Left neighbor
        if patch_col > 0:
            left_neighbor = patch_row * patches_per_row + patch_col - 1
            self.counter[left_neighbor] = (self.counter[left_neighbor] + left_edge) / 2
    
    def stitch_patches(self, patch_tensors):
        patches_per_side = self.image_size // self.patch_size
        full_image = torch.zeros(self.image_size, self.image_size)
        for i in range(patches_per_side):
            for j in range(patches_per_side):
                patch_idx = i * patches_per_side + j
                if patch_idx < len(patch_tensors):
                    start_i = i * self.patch_size
                    start_j = j * self.patch_size
                    patch = patch_tensors[patch_idx].view(self.patch_size, self.patch_size)
                    full_image[start_i:start_i+self.patch_size, start_j:start_j+self.patch_size] = patch
        
        return full_image

def quantum_generator_with_patches(quantum_generator, generator_params, latent_dim, n_patches, patch_correlation_counter):
    patch_tensors = []
    for i in range(n_patches):
        latent_vector = torch.randn(latent_dim) * 0.05
        boundary_condition = patch_correlation_counter.get_boundary_condition(i)
        latent_vector_np = latent_vector.detach().numpy()
        if len(latent_vector_np) >= 4:
            latent_vector_np[:4] = latent_vector_np[:4] * 0.5 + boundary_condition
        
        try:
            params_np = generator_params.detach().numpy()
            quantum_output = quantum_generator(latent_vector_np, params_np)
            patch_tensor = torch.tensor(quantum_output, dtype=torch.float32)
            patch_size = patch_correlation_counter.patch_size
            patch_pixels = quantum_generator_output_to_image(patch_tensor, patch_size)
            patch_correlation_counter.update_counter(i, patch_pixels)
            patch_tensors.append(patch_pixels)
        except Exception as e:
            print(f"Error generating patch {i}: {str(e)}")
            patch_size = patch_correlation_counter.patch_size
            patch_pixels = torch.rand(patch_size * patch_size)
            patch_correlation_counter.update_counter(i, patch_pixels)
            patch_tensors.append(patch_pixels)
    
    full_image = patch_correlation_counter.stitch_patches(patch_tensors)
    return full_image

def train_quantum_gan(quantum_generator, n_params, train_loader):
    critic = ClassicalCritic(image_size * image_size)
    generator_params = nn.Parameter(torch.zeros(n_params) + 0.01)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
    generator_optimizer = optim.Adam([generator_params], lr=lr_generator)
    patch_size = 7  # Each patch will be 7x7
    n_patches = (image_size // patch_size) ** 2
    patch_correlation_counter = PatchCorrelationCounter(image_size, patch_size)
    
    def fallback_generator(batch_size):
        return torch.rand(batch_size, num_qubits) * 2 - 1
    
    for epoch in range(n_epochs):
        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            real_batch_size = real_imgs.size(0)
            real_imgs = real_imgs.view(real_batch_size, -1)
            
            quantum_batch_size = min(8, real_batch_size)
            # Training Critic
            for _ in range(critic_iterations):
                critic_optimizer.zero_grad()
                
                fake_imgs = []
                for i in range(quantum_batch_size):
                    patch_correlation_counter.reset()
                    fake_img = quantum_generator_with_patches(
                        quantum_generator, 
                        generator_params, 
                        latent_dim, 
                        n_patches, 
                        patch_correlation_counter
                    )
                    fake_imgs.append(fake_img.flatten())
                
                fake_imgs = torch.stack(fake_imgs)
                
                if fake_imgs.size(0) < real_batch_size:
                    repeat_factor = (real_batch_size + fake_imgs.size(0) - 1) // fake_imgs.size(0)
                    fake_imgs = fake_imgs.repeat(repeat_factor, 1)
                    fake_imgs = fake_imgs[:real_batch_size]
                
                real_validity = critic(real_imgs)
                fake_validity = critic(fake_imgs)                
                gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs)
                # Wasserstein loss with gradient penalty
                critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic_optimizer.step()
            
            generator_optimizer.zero_grad()
            
            fake_imgs = []
            for i in range(quantum_batch_size):
                patch_correlation_counter.reset()
                fake_img = quantum_generator_with_patches(
                    quantum_generator, 
                    generator_params, 
                    latent_dim, 
                    n_patches, 
                    patch_correlation_counter
                )
                fake_imgs.append(fake_img.flatten())
            
            fake_imgs = torch.stack(fake_imgs)
            
            if fake_imgs.size(0) < real_batch_size:
                repeat_factor = (real_batch_size + fake_imgs.size(0) - 1) // fake_imgs.size(0)
                fake_imgs = fake_imgs.repeat(repeat_factor, 1)
                fake_imgs = fake_imgs[:real_batch_size]
            
            fake_validity = critic(fake_imgs)
            generator_loss = -torch.mean(fake_validity)
            
            if torch.isnan(generator_loss):
                print("Skipping generator backpropagation due to NaN loss")
                continue
            
            # Compute additional metrics for RL feedback
            try:
                inception_score = compute_inception_score(fake_imgs)
                generator_loss = generator_loss - 0.1 * torch.tensor(inception_score)
            except Exception as e:
                print(f"Error computing inception score: {str(e)}")
            
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_([generator_params], max_norm=0.5)
            generator_optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {batch_idx}/{len(train_loader)}] "
                      f"[Critic loss: {critic_loss.item():.4f}] [Generator loss: {generator_loss.item():.4f}]")
                
                # Periodically save an image
                if batch_idx % 50 == 0:
                    try:
                        plt.figure(figsize=(5, 5))
                        plt.imshow(fake_imgs[0].view(image_size, image_size).detach().numpy(), cmap='gray')
                        plt.savefig(f'sample_epoch{epoch}_batch{batch_idx}.png')
                        plt.close()
                        
                        plt.figure(figsize=(5, 5))
                        img_with_boundaries = fake_imgs[0].view(image_size, image_size).detach().numpy()
                        for i in range(0, image_size, patch_size):
                            img_with_boundaries[i, :] = 1.0
                            img_with_boundaries[:, i] = 1.0
                        plt.imshow(img_with_boundaries, cmap='gray')
                        plt.savefig(f'sample_with_boundaries_epoch{epoch}_batch{batch_idx}.png')
                        plt.close()
                    except Exception as e:
                        print(f"Error saving sample image: {str(e)}")
    
    return generator_params

def main():
    print("Starting Adaptive Quantum Circuit GAN with RL Optimization")
    # Step 1: Train RL agent to design an optimal quantum circuit
    print("Phase 1: Training RL agent for circuit design optimization...")
    best_circuit_design = train_rl_agent_to_design_circuit(max_iterations=500)
    # Step 2: Create quantum generator from the designed circuit
    print("Phase 2: Creating quantum generator from optimized circuit design...")
    quantum_generator, n_params = create_quantum_generator_from_design(best_circuit_design)
    # Step 3: Load dataset
    print("Phase 3: Loading MNIST dataset...")
    train_loader = load_mnist_data()
    # Step 4: Train the quantum GAN
    print("Phase 4: Training quantum GAN with the optimized circuit...")
    trained_params = train_quantum_gan(quantum_generator, n_params, train_loader)
    # Save the trained model
    torch.save({
        'generator_params': trained_params,
        'circuit_design': best_circuit_design
    }, 'quantum_gan_model.pth')
    
    print("Training complete. Model saved to quantum_gan_model.pth")

if __name__ == "__main__":
    main()

