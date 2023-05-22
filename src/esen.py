#
# Esen model: irregular spiking neural networks
#
# May '23

from brian2 import *
import numpy as np

def generate_matrix(N, alpha=1.5, sigma=0.1):
    A = np.random.normal(loc=0, scale=sigma, size=(N, N))
    A = 0.5 * (A + A.T)  # make the matrix symmetric
    eigvals, eigvecs = np.linalg.eigh(A)  # get eigenvalues and eigenvectors
    eigvals.sort()  # sort eigenvalues in ascending order
    eigvals[-1] = eigvals[-2] * alpha  # set largest eigenvalue to alpha times the first eigenvalue
    A = eigvecs @ np.diag(eigvals) @ eigvecs.T  # reconstruct the matrix
    return A

def GenerateWigner(N, mu=0, sigma=1):
    # Wigner weight matrix
    W = np.random.normal(mu, sigma, size=(N, N))
    W = (W + W.T) / 2 # symmetrize the matrix
    W -= np.diag(np.diag(W)) # set diagonal entries to zero
    return W

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
#W = GenerateMatrix(N)
#W = GenerateWigner(N)
W = generate_matrix(N, alpha=1)
# W = (W + W.T) / 2
# W -= np.diag(np.diag(W))    # set diagonal entries to zero
assert(np.allclose(W, W.T))

eigvals = np.linalg.eigvals(W)
#print(eigvals)
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
#S.w = np.concatenate((W[idx1], W[idx2]))
S.w = 0.5*np.ones((N*(N-1)))
              
# Simulate network again and record spike times
M = SpikeMonitor(G)
LFP = PopulationRateMonitor(G)
selected_neurons = [0] #, 100, 200, 300, 400, 500, 600, 700, 800, 900]
Monitors = StateMonitor(G, 'V', record=selected_neurons)
net = Network(G, S, M, LFP, Monitors)
#w_monitor = StateMonitor(S, 'w', record=True)
#net = Network(G, S, M, LFP, Monitors, w_monitor)

# Run
net.run(duration)
#muext = 30*mV
#net.run(20*ms)
#muext = 0*mV
#net.run(duration-20*ms)

# Plot the value of S.w over time
""" figure()
plot(w_monitor.w)
print(w_monitor.w)
xlabel('Time (ms)')
ylabel('Synaptic weight (w)')
show() """

# Plot the histogram of firing frequency
# Get the firing rates
firing_rates = M.count / duration
# Plot the histogram
figure()
hist(firing_rates/Hz, bins=50)
xlabel('Firing rate (sp/s)')
ylabel('Number of neurons')
show()

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

