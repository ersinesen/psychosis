# Psychosis

Computational models of psychosis

## Special Matrix

* Wigner

```
mu, sigma = 0, 1 # mean and standard deviation of Gaussian distribution
W = np.random.normal(mu, sigma, size=(N, N))
W = (W2 + W2.T) / 2 # symmetrize the matrix
W -= np.diag(np.diag(W2)) # set diagonal entries to zero
```

## Notes

### When external input is cut

ChatGPT: Here are a few expected behaviors after the external input is cut:

Spontaneous Activity: The network will exhibit spontaneous, self-generated activity in the absence of external input. This activity arises from the recurrent connections between neurons and their intrinsic dynamics. Neurons may continue to spike and interact with each other, generating complex patterns of activity.

Network Oscillations: Depending on the specific connectivity and synaptic weights, the network may exhibit oscillatory behavior. Oscillations can emerge due to the interactions and feedback between neurons in the network. These oscillations can manifest as rhythmic patterns in the spike activity of the neurons.

Adaptation: Neurons may exhibit adaptation, where their firing rates or patterns change over time. This adaptation can occur as a result of the network's dynamics and the synaptic plasticity mechanisms at play. Neurons may adjust their firing rates or become more or less responsive to input based on the network's ongoing activity.

Stability or Transient Dynamics: The network's activity may stabilize into a steady state where the firing rates and patterns remain relatively constant. Alternatively, the network may exhibit transient dynamics, where the activity undergoes changes or evolves over time before settling into a stable state. These dynamics are influenced by the initial conditions, connectivity, and synaptic strengths of the network.

[Source](https://chat.openai.com/c/ce7b89f5-9095-4664-8801-38eeea7e9a02)


## Links

* [Emergent stability in complex network dynamics](https://www.nature.com/articles/s41567-023-02020-8)

* [Power (von Mises) Iteration](https://en.wikipedia.org/wiki/Power_iteration)

* [Brunel and Hakim Model](https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Hakim_1999.html)

* [ChatpGPT: Brunel and Hakim Model](https://chat.openai.com/c/30aacb02-d06b-4bbf-99ba-e519d641f404)

