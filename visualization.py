import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

num_qubits = 8
latent_dim = 16
image_size = 28  # For MNIST
GATE_OPTIONS = {
    0: "RX",
    1: "RY",
    2: "RZ",
    3: "Hadamard",
    4: "CNOT"
}

dev = qml.device("default.qubit", wires=num_qubits, shots=None)

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
                    self.parameters.append(None)  # No parameter for CNOT
            else:
                self.gates[self.current_layer].append((gate_type, target_qubit))
                self.qubit_usage[target_qubit] += 1
                self.parameters.append(None)  # Placeholder
                
    def next_layer(self):
        if self.current_layer < self.max_layers:
            self.current_layer += 1
            return True
        return False

def dynamic_quantum_generator(latent_vector, circuit_design, params):
    latent_vector = np.tanh(latent_vector) * 0.5
    for i, value in enumerate(latent_vector):
        if i < num_qubits:
            qml.RY(value * np.pi, wires=i)
    param_idx = 0
    for layer_idx, layer in enumerate(circuit_design.gates):
        if not layer:
            continue
            
        for gate in layer:
            gate_type = gate[0]
            if param_idx < len(params):
                param_value = np.tanh(params[param_idx].item()) * np.pi
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

def create_quantum_generator_from_design(circuit_design):
    n_params = sum(1 for layer in circuit_design.gates for gate in layer 
                   if gate[0] in [0, 1, 2])  # RX, RY, RZ
    n_params += num_qubits
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_generator(latent_vector, params):
        return dynamic_quantum_generator(latent_vector, circuit_design, params)
    
    return quantum_generator, n_params

def quantum_generator_output_to_image(quantum_output, image_size):
    scaled_output = (quantum_output + 1) / 2
    reshaped_output = torch.zeros(image_size * image_size)
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

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
    model_data = torch.load(model_path, weights_only=False)
    circuit_design = model_data['circuit_design']
    generator_params = model_data['generator_params']
    
    return circuit_design, generator_params

def generate_images(quantum_generator, generator_params, n_images=10, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    latent_vectors = torch.randn(n_images, latent_dim) * 0.05
    params_np = generator_params.detach().numpy()
    generated_images = []
    for i in range(n_images):
        try:
            quantum_output = quantum_generator(latent_vectors[i].numpy(), params_np)
            image_tensor = quantum_generator_output_to_image(torch.tensor(quantum_output), image_size)
            image = image_tensor.view(image_size, image_size).detach().numpy()
            generated_images.append(image)
        except Exception as e:
            print(f"Error generating image {i}: {str(e)}")
    
    return generated_images

def plot_generated_images(images, rows=2, cols=5, figsize=(15, 6)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_circuit_design(circuit_design):
    plt.figure(figsize=(14, 8))
    gate_counts = {gate_name: 0 for gate_name in GATE_OPTIONS.values()}
    connections = []
    for layer_idx, layer in enumerate(circuit_design.gates):
        for gate in layer:
            gate_type = gate[0]
            gate_name = GATE_OPTIONS[gate_type]
            gate_counts[gate_name] += 1
            
            if gate_type == 4 and len(gate) > 2:  # CNOT
                connections.append((gate[1], gate[2])) 
    
    total_gates = sum(gate_counts.values())
    print(f"Total gates: {total_gates}")
    for gate_name, count in gate_counts.items():
        print(f"  - {gate_name}: {count} gates ({count/max(1, total_gates)*100:.1f}%)")
    
    print("\nQubit Connectivity:")
    connectivity = circuit_design.connectivity_matrix
    for i in range(circuit_design.num_qubits):
        for j in range(circuit_design.num_qubits):
            if connectivity[i,j] > 0:
                print(f"  - Qubit {i} -> Qubit {j}: {int(connectivity[i,j])} connections")
    
    plt.subplot(2, 2, 1)
    plt.bar(gate_counts.keys(), gate_counts.values())
    plt.title('Gate Type Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.tight_layout()
    
    plt.subplot(2, 2, 2)
    sns.heatmap(circuit_design.qubit_usage.reshape(-1, 1), 
                annot=True, fmt='.1f', cmap='Blues',
                yticklabels=[f'Q{i}' for i in range(circuit_design.num_qubits)],
                xticklabels=['Usage'])
    plt.title('Qubit Usage')
    plt.tight_layout()
    
    plt.subplot(2, 2, 3)
    connectivity_matrix = circuit_design.connectivity_matrix
    sns.heatmap(connectivity_matrix, annot=True, fmt='.0f', cmap='viridis',
                xticklabels=[f'Q{i}' for i in range(circuit_design.num_qubits)],
                yticklabels=[f'Q{i}' for i in range(circuit_design.num_qubits)])
    plt.title('Qubit Connectivity Matrix')
    plt.xlabel('Target Qubit')
    plt.ylabel('Control Qubit')
    plt.tight_layout()
    
    plt.subplot(2, 2, 4)
    G = nx.DiGraph()
    
    for i in range(circuit_design.num_qubits):
        G.add_node(i, label=f'Q{i}')
    
    for source, target in connections:
        G.add_edge(source, target, weight=connectivity_matrix[source, target])
    
    pos = nx.circular_layout(G)    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels={i: f'Q{i}' for i in range(circuit_design.num_qubits)})
    
    for (u, v, d) in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=d['weight'], 
                               alpha=0.7, arrows=True, arrowsize=15)
    
    plt.axis('off')
    plt.title('Circuit Connectivity Graph')
    plt.tight_layout()    
    plt.savefig('circuit_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_quantum_circuit(circuit_design, generator_params):
    print("Creating immediate ASCII circuit visualization...")
    ascii_circuit = []
    header = "Qubits:"
    for i in range(num_qubits):
        header += f" q{i:<3}"
    ascii_circuit.append(header)
    
    separator = "=" * (7 + 4 * num_qubits)
    ascii_circuit.append(separator)
    
    qubit_positions = [0] * num_qubits
    max_layer_idx = 0
    for layer_idx, layer in enumerate(circuit_design.gates):
        if not layer:
            continue
            
        max_layer_idx = layer_idx
        
        layer_lines = [""] * num_qubits
        for i in range(num_qubits):
            layer_lines[i] = "    " * qubit_positions[i] + "|   " 
        
        for gate in layer:
            gate_type = gate[0]
            
            if gate_type in [0, 1, 2, 3]:  # Single-qubit gates
                qubit = gate[1]
                if qubit < num_qubits:  # Ensure qubit is in range
                    gate_symbol = ["RX", "RY", "RZ", "H"][gate_type]
                    layer_lines[qubit] = layer_lines[qubit][:-4] + f"{gate_symbol:<4}"
                    qubit_positions[qubit] += 1
            
            elif gate_type == 4 and len(gate) > 2:  # CNOT
                control = gate[1]
                target = gate[2]
                if control < num_qubits and target < num_qubits:
                    layer_lines[control] = layer_lines[control][:-4] + "C   " 
                    qubit_positions[control] += 1                    
                    layer_lines[target] = layer_lines[target][:-4] + "X   " 
                    qubit_positions[target] += 1
        
        for i in range(num_qubits):
            ascii_circuit.append(f"q{i:<2} | {layer_lines[i]}")        
        ascii_circuit.append("-" * (7 + 4 * max(qubit_positions)))
    
    with open('quantum_circuit_ascii.txt', 'w', encoding='utf-8') as f:
        for line in ascii_circuit:
            f.write(line + "\n")
    
    print("\nCircuit Preview (first 20 lines):")
    print("---------------------------------")
    for line in ascii_circuit[:min(20, len(ascii_circuit))]:
        print(line)
    print("\nFull circuit saved to 'quantum_circuit_ascii.txt'")
    
    try:
        qubit_gate_counts = [0] * num_qubits
        connections = []
        
        for layer in circuit_design.gates:
            if not layer:
                continue
                
            for gate in layer:
                gate_type = gate[0]
                if gate_type in [0, 1, 2, 3]:  # Single-qubit gates
                    qubit = gate[1]
                    if qubit < num_qubits:
                        qubit_gate_counts[qubit] += 1
                elif gate_type == 4 and len(gate) > 2:  # CNOT
                    control = gate[1]
                    target = gate[2]
                    if control < num_qubits and target < num_qubits:
                        connections.append((control, target))
                        qubit_gate_counts[control] += 1
                        qubit_gate_counts[target] += 1
        
        fig, ax = plt.subplots(figsize=(10, num_qubits * 0.5 + 2))        
        for i in range(num_qubits):
            ax.plot([0, 1], [i, i], 'k-', linewidth=2)
            ax.text(-0.05, i, f'q{i}', horizontalalignment='right', verticalalignment='center')
        
        for i in range(num_qubits):
            ax.text(1.05, i, f'{qubit_gate_counts[i]} gates', 
                    horizontalalignment='left', verticalalignment='center')        
        connection_counts = {}
        for control, target in connections:
            key = (min(control, target), max(control, target))
            connection_counts[key] = connection_counts.get(key, 0) + 1
        
        top_connections = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        for (qubit1, qubit2), count in top_connections:
            opacity = min(1.0, 0.2 + count / max(connection_counts.values()))
            width = 0.5 + count / max(connection_counts.values()) * 2
            ax.plot([0.5, 0.5], [qubit1, qubit2], 'r-', alpha=opacity, linewidth=width)
            ax.text(0.52, (qubit1 + qubit2) / 2, f'{count}', 
                    color='red', alpha=opacity,
                    horizontalalignment='left', verticalalignment='center')
        
        ax.set_xlim(-0.1, 1.5)
        ax.set_ylim(-0.5, num_qubits - 0.5)
        ax.set_title(f'Simplified Circuit Overview\n{sum(qubit_gate_counts)} total gates, {max_layer_idx+1} layers')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('quantum_circuit_simple.png', dpi=100)
        plt.close()
        
        print("Simplified circuit visual created: 'quantum_circuit_simple.png'")
        
        return True
    except Exception as e:
        print(f"Error creating simplified circuit visual: {str(e)}")
        return False

def plot_parameter_heatmap(generator_params):
    params_np = generator_params.detach().numpy()
    n_params = len(params_np)
    n_rows = max(1, n_params // 8)
    n_cols = min(8, n_params)
    padding = n_rows * n_cols - n_params
    if padding > 0:
        params_np = np.pad(params_np, (0, padding), mode='constant', constant_values=np.nan)
    params_reshaped = params_np[:n_rows*n_cols].reshape(n_rows, n_cols)
    plt.figure(figsize=(12, max(4, n_rows)))
    cmap = LinearSegmentedColormap.from_list('custom_diverge', 
                                            [(0, 'blue'), (0.5, 'white'), (1.0, 'red')], 
                                            N=256)
    
    ax = sns.heatmap(params_reshaped, cmap=cmap, center=0, 
                    annot=True, fmt='.3f', 
                    mask=np.isnan(params_reshaped))
    
    plt.title('Optimized Generator Parameters', fontsize=16)
    plt.xlabel('Parameter Index', fontsize=12)
    plt.ylabel('Parameter Group', fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(ax.collections[0], cax=cax)
    
    plt.tight_layout()
    plt.savefig('parameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_latent_space_exploration(quantum_generator, generator_params, n_samples=10):
    np.random.seed(42)
    torch.manual_seed(42)
    base_latent = np.zeros(latent_dim)
    variations = np.linspace(-0.2, 0.2, n_samples)
    dim1, dim2 = np.random.choice(min(latent_dim, 5), 2, replace=False)
    grid_images = []
    params_np = generator_params.detach().numpy()
    
    for v1 in variations:
        row_images = []
        for v2 in variations:
            latent = base_latent.copy()
            latent[dim1] = v1
            latent[dim2] = v2
            
            try:
                quantum_output = quantum_generator(latent, params_np)
                image_tensor = quantum_generator_output_to_image(torch.tensor(quantum_output), image_size)
                image = image_tensor.view(image_size, image_size).detach().numpy()
                row_images.append(image)
            except Exception as e:
                print(f"Error exploring latent space at ({v1}, {v2}): {str(e)}")
                row_images.append(np.zeros((image_size, image_size)))
        
        grid_images.append(row_images)
    
    fig, axes = plt.subplots(n_samples, n_samples, figsize=(12, 12))
    
    for i in range(n_samples):
        for j in range(n_samples):
            axes[i, j].imshow(grid_images[i][j], cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle(f'Latent Space Exploration\nVarying dimensions {dim1} and {dim2}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('latent_space_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Quantum GAN Visualization Tool")
    model_path = 'quantum_gan_model.pth'
    try:
        print(f"Loading model from {model_path}...")
        circuit_design, generator_params = load_model(model_path)
        print("Creating quantum generator from design...")
        quantum_generator, n_params = create_quantum_generator_from_design(circuit_design)
        print("Creating immediate circuit visualization...")
        draw_quantum_circuit(circuit_design, generator_params)
        print("Visualizing parameters...")
        plot_parameter_heatmap(generator_params)
        # Generate sample images
        n_images = 4
        print(f"Generating {n_images} images...")
        generated_images = generate_images(quantum_generator, generator_params, n_images=n_images)
        
        print("Plotting generated images...")
        plot_generated_images(generated_images, rows=2, cols=2)
        
        print("Visualization complete! ASCII circuit created, simplified visual generated, and sample images displayed.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
