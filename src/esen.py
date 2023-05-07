#
# Esen model: irregular spiking neural networks
#
# May '23

from brian2 import *

def GenerateMatrix(N):
    # Generate a random matrix
    R = np.random.randn(N, N)

    # Compute the SVD of R and set U = V
    U, _, Vt = np.linalg.svd(R)
    V = Vt.T

    # Generate a vector of eigenvalues with two close largest values
    w = np.random.normal(loc=0, scale=1, size=N)
    w[0] = 1
    w[1] = 0.99

    # Construct the diagonal matrix with the eigenvalues
    D = np.diag(w)

    # Compute the weight matrix
    W = U @ D @ V.T
    
    return W

# Parameters
N = 1000
duration = .2*second

# Neuron model
Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
C = 100
sparseness = float(C)/N
J = .1*mV
muext = 30*mV
sigmaext = 1*mV
eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""

# Weigth matrix with equal largest 2 eigenvalues
W = GenerateMatrix(N)

eigvals = np.linalg.eigvals(W)
print("eigenvalues: ", eigvals[:10])

# Network
G = NeuronGroup(N, eqs, threshold='V>theta', reset='V=Vr', refractory=taurefr, method='euler')
G.V = Vr
S = Synapses(G, G, 'w : 1', on_pre='V += -J*w', delay=delta)
# fully connected
S.connect(condition='i!=j')
# set the weights
idx1 = np.triu_indices(N, k=1)
idx2 = np.tril_indices(N, k=-1)
S.w = np.concatenate((W[idx1], W[idx2]))

# Simulate network again and record spike times
M = SpikeMonitor(G)
LFP = PopulationRateMonitor(G)
selected_neurons = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
Monitors = StateMonitor(G, 'V', record=selected_neurons)
net = Network(G, S, M, LFP, Monitors)

# Run
net.run(duration)
#muext = 30*mV
#net.run(20*ms)
#muext = 0*mV
#net.run(duration-20*ms)

# Plot the results
figure()
subplot(211)
plot(M.t/ms, M.i, '.')
xlim(0, duration/ms)
xlabel('Neuron Index')
ylabel('Time (ms)')

subplot(212)
plot(LFP.t/ms, LFP.smooth_rate(window='flat', width=0.5*ms)/Hz)
xlim(0, duration/ms)
xlabel('Firing Neuron Count')
ylabel('Time (ms)')

figure()
for i in range(len(selected_neurons)):
    plot(Monitors.t/ms, Monitors.V[i], label=f'Neuron {selected_neurons[i]}')
xlabel('Time (ms)')
ylabel('Voltage')
legend()


show()

