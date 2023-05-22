#
# Oscillation after input is cut
#
# EE & ChatGPT May '23

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

import numpy as np

def create_adjusted_connectivity(num_neurons):
    # Create random binary matrix
    connectivity = np.random.randint(2, size=(num_neurons, num_neurons))
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(connectivity)
    
    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    
    # Adjust the eigenvalues
    sorted_eigenvalues[0] = 1*sorted_eigenvalues[1]
    
    # Reconstruct the adjusted connectivity matrix
    adjusted_connectivity = np.dot(eigenvectors[:, sorted_indices], np.diag(sorted_eigenvalues)).dot(np.linalg.inv(eigenvectors[:, sorted_indices]))
    
    # Threshold the matrix to have binary elements (0 or 1)
    adjusted_connectivity = np.where(adjusted_connectivity >= 0.5, 1, 0)
    
    return adjusted_connectivity

def GenerateMatrix(N):
    # Generate a random matrix
    R = np.random.randn(N, N)

    # Compute the SVD of R and set U = V
    U, _, Vt = np.linalg.svd(R)
    V = Vt.T

    # Generate a vector of eigenvalues with two close largest values
    w = np.random.normal(loc=0, scale=1, size=N)
    w[0] = 1
    w[1] = 0.9

    # Construct the diagonal matrix with the eigenvalues
    D = np.diag(w)

    # Compute the weight matrix
    W = U @ D @ V.T
    
    return W


# Set random seed for reproducibility
np.random.seed(123)

# Parameters
num_neurons = 100  # Number of neurons in the network
duration = 1000 * ms  # Total simulation time
input_duration = 100 * ms  # Duration of external input
refractory_period = 10 * ms  # Refractory period duration

# Model equations
tau = 10 * ms  # Membrane time constant
eqs = '''
dv/dt = (-v + I) / tau : 1
dI/dt = -I / tau : 1
'''

# Create neuron group
neurons = NeuronGroup(num_neurons, eqs, threshold='v>0.9', reset='v=0', refractory=refractory_period)

# Set initial conditions for the neurons
neurons.v = 'rand()'
neurons.I = 0

# Randomly set up connectivity using adjacency matrix
connectivity = np.random.rand(num_neurons, num_neurons) < 0.03  # 10% connectivity probability
#connectivity = create_adjusted_connectivity(num_neurons)
print("Number of Connections: ", np.sum(connectivity))

# condition number
cond_number = np.linalg.cond(connectivity)
print("Condition number of connectivity:", cond_number)

# eigenvalu ratio
eigenvalues, eigenvectors = np.linalg.eig(connectivity)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
print("ev1/ev2:", np.abs(sorted_eigenvalues[0]/sorted_eigenvalues[1]))

S = Synapses(neurons, neurons, 'w : 1', on_pre='I_post += w')
i, j = np.where(connectivity)
S.connect(i=i, j=j)

# Initialize synaptic weights
#W = GenerateMatrix(num_neurons)
#S.w = W[connectivity==1]
S.w = 'rand()'

# Create external input
input_neurons = PoissonGroup(num_neurons, rates=10 * Hz)  # Poisson spiking input neurons
input_syn = Synapses(input_neurons, neurons, 'w : 1', on_pre='I_post += w')
input_syn.connect(p=0.2)  # 5% connectivity probability for input synapses

# Initialize synaptic weights for input synapses
input_syn.w = '1' #'rand()'

# Record spikes
spike_mon = SpikeMonitor(neurons)

# Run simulation with external input
run(input_duration)

# Cease external input
input_neurons.rates = 0 * Hz  # Set input rates to zero

# Continue simulation without external input
run(duration - input_duration)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(spike_mon.t/ms, spike_mon.i, 'k.', label='Neuron spikes')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike raster plot')
plt.legend()
plt.show()
