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



## Links

* [Emergent stability in complex network dynamics](https://www.nature.com/articles/s41567-023-02020-8)

* [Power (von Mises) Iteration](https://en.wikipedia.org/wiki/Power_iteration)

* [Brunel and Hakim Model](https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Hakim_1999.html)

* [ChatpGPT: Brunel and Hakim Model](https://chat.openai.com/c/30aacb02-d06b-4bbf-99ba-e519d641f404)

