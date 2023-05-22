from brian2 import *

# Set random seed for reproducibility
np.random.seed(42)

# simulation
duration = 1000*ms

# Define network parameters
num_input_neurons = 5
num_intermediate_neurons = 20
num_output_neurons = 5

# Define neuron model
tau = 10*ms
eqs = '''
dv/dt = (-v + I)/tau : volt (unless refractory)
I : volt
'''

# Create input neurons with random initial currents
input_neurons = NeuronGroup(num_input_neurons, eqs, threshold='v > 1*mV',
                            reset='v = 0*mV', refractory=5*ms)
input_neurons.I = 20*mV #np.random.uniform(low=-1*mV, high=1*mV, size=num_input_neurons) *mV

# Create intermediate neurons
intermediate_neurons = NeuronGroup(num_intermediate_neurons, eqs, threshold='v > 1*mV',
                                   reset='v = 0*mV', refractory=5*ms)

# Create output neurons
output_neurons = NeuronGroup(num_output_neurons, eqs, threshold='v > 1*mV',
                             reset='v = 0*mV', refractory=5*ms)

# Connect input neurons to intermediate neurons
input_synapses = Synapses(input_neurons, intermediate_neurons, on_pre='v += 0.1*mV')
input_synapses.connect()

# Connect intermediate neurons to output neurons
output_synapses = Synapses(intermediate_neurons, output_neurons, on_pre='v += 0.1*mV')
output_synapses.connect()

# Connect output neurons to input neurons
feedback_synapses = Synapses(output_neurons, input_neurons, on_pre='v += 0.1*mV')
feedback_synapses.connect()

# Create adjacency matrix for intermediate neurons
adjacency_matrix = np.random.randint(2, size=(num_intermediate_neurons, num_intermediate_neurons))

# Connect intermediate neurons using adjacency matrix
intermediate_synapses = Synapses(intermediate_neurons, intermediate_neurons,
                                 'w : volt', on_pre='v_post += w')
intermediate_synapses.connect(i=np.where(adjacency_matrix==1)[0],
                               j=np.where(adjacency_matrix==1)[1])
intermediate_synapses.w = '0.1*mV*rand()'

# Define monitor
M = SpikeMonitor(intermediate_neurons)

# Run simulation
#run(100*ms)
run(80*ms)
input_neurons.I = 0*mV
run(duration-120*ms)


#print(M.num_spikes, M.i)

# Plot spike raster
plot(M.t/ms, M.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
show()
