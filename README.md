# Diffusion Models in Graph-Laplacian Latent Space

First experiments on training diffusion models in a latent space inferred from the graph-Laplacian eigenfunctions. The graph is constructed from a subset of the data, and the eigenfunctions provide a compact embedding for the data. The distribution of the coefficients to represent the data shows useful properties and is easy to learn. The encoder-decoder models are defined with a linear (non-trainable) map as backbone, based on the Laplacian eigenfunctions, and a shallow, trainable CNN-based head. 

