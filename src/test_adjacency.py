import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2

# Adjacency Matrix transformation
def Transform(A):
    # Set the number of iterations to perform
    num_iters = 100

    # Perform the iterations
    for i in range(num_iters):
        # Calculate the eigenvalues of the matrix
        eigvals = np.linalg.eigvals(A)

        # Find the two eigenvalues with the largest norms
        max_eigvals = np.sort(np.abs(eigvals))[-2:]
        idx = np.argsort(np.abs(eigvals))[::-1][:2]
        print(max_eigvals, idx)

        # Check if the norms of the two eigenvalues are equal
        if np.abs(np.abs(max_eigvals[0]) - np.abs(max_eigvals[1])) < 1e-6:
            break

        # Modify the adjacency matrix to make the norms of the largest two eigenvalues equal
        a = np.abs(max_eigvals[0])
        b = np.random.uniform(low=0, high=2*np.pi)
        c = a * np.exp(1j*b)
        d = a * np.exp(-1j*b)
        A = c * np.outer(np.abs(max_eigvals[0])**(-1/2) * np.real(eigvals[:, idx[0]]),
                        np.abs(max_eigvals[0])**(-1/2) * np.real(eigvals[:, idx[0]]).T) \
            + d * np.outer(np.abs(max_eigvals[1])**(-1/2) * np.real(eigvals[:, idx[1]]),
                        np.abs(max_eigvals[1])**(-1/2) * np.real(eigvals[:, idx[1]]).T)

    # Check if the iterations converged
    if i == num_iters-1:
        print("Warning: maximum number of iterations reached without convergence")

    # Print the eigenvalues of the final matrix
    print(np.linalg.eigvals(A))
    
    return A
 

# define number of neurons and connection rate
num_neurons = 100
connection_rate = 0.1

# generate random adjacency matrix
adj_matrix = np.zeros((num_neurons, num_neurons))
for i in range(num_neurons):
    for j in range(num_neurons):
        if np.random.rand() < connection_rate:
            adj_matrix[i, j] = 1

# Generate a random initial adjacency matrix
#adj_matrix = np.random.randint(0, 2, size=(N, N))

# Set the diagonal elements to zero
np.fill_diagonal(adj_matrix, 0)
print(adj_matrix.shape)
adj_matrix = Transform(adj_matrix)

# define neuron model
tau = 10 * b2.ms
eqs = '''
dv/dt = (I_syn + I_ext) / tau : 1
I_syn : 1 
I_ext : 1
'''

# create neuron group and connections
neuron_group = b2.NeuronGroup(num_neurons, eqs, threshold='v>1', reset='v=0', method='rk4')
neuron_group.I_ext = np.random.rand(num_neurons) 
neuron_group.I_ext *= 0.2

synapses = b2.Synapses(neuron_group, neuron_group, 'w : 1', on_pre='I_syn += w')
for i in range(num_neurons):
    for j in range(num_neurons):
        if adj_matrix[i][j] != 0:
            synapses.connect(i=i, j=j)
            synapses.w[i, j] = adj_matrix[i][j]
synapses.w = 'rand()'

#synapses = b2.Synapses(neuron_group, neuron_group, 'w : 1', on_pre='v += w')
#synapses.connect(condition='i!=j', p=0.2)  # connect with probability 0.2
#synapses.w = 'randn()*0.1'  # set initial weights to random values

# record voltage of selected neurons
selected_neurons = [0, 10, 20]
monitors = b2.StateMonitor(neuron_group, 'v', record=selected_neurons)

# run simulation
sim_duration = 1000 * b2.ms
b2.run(sim_duration, report='text')

# plot voltage of selected neurons
plt.figure()
for i in range(len(selected_neurons)):
    plt.plot(monitors.t/b2.ms, monitors.v[i], label=f'Neuron {selected_neurons[i]}')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage')
plt.legend()
plt.show()
