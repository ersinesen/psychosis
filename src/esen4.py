#
# Two groups oscillating after input is cut
#
# EE & ChatGPT May '23

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# Set random seed for reproducibility
np.random.seed(123)

# Parameters
num_neurons = 200  # Total number of neurons
group_size = num_neurons // 2  # Size of each group
duration = 1000 * ms  # Total simulation time
input_duration = 500 * ms  # Duration of external input
refractory_period = 5 * ms  # Refractory period duration

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

# Divide neurons into two groups
group_A = neurons[:group_size]
group_B = neurons[group_size:]

# Set up connectivity within each group
connectivity_within_group = np.random.rand(group_size, group_size) < 0.8  # Connectivity probability within each group
i, j = np.where(connectivity_within_group)
S_within_group = Synapses(group_A, group_A, 'w : 1', on_pre='I_post += w')
S_within_group.connect(i=i, j=j)
S_within_group.w = 'rand()'

S_within_group2 = Synapses(group_B, group_B, 'w : 1', on_pre='I_post += w')
S_within_group2.connect(i=i, j=j)
S_within_group2.w = 'rand()'

# Set up connectivity between groups
connectivity_between_groups = np.random.rand(group_size, group_size) < 0.2  # Connectivity probability between groups
i, j = np.where(connectivity_between_groups)
S_between_groups = Synapses(group_A, group_B, 'w : 1', on_pre='I_post += w')
S_between_groups.connect(i=i, j=j)
S_between_groups.w = 'rand()'

# Create external input
input_neurons = PoissonGroup(num_neurons, rates=10 * Hz)  # Poisson spiking input neurons
input_syn = Synapses(input_neurons, neurons, 'w : 1', on_pre='I_post += w')
input_syn.connect(p=0.1)  # 10% connectivity probability for input synapses
input_syn.w = 'rand()'

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
