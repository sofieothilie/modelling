# Project description

This project is a CUDA implementation of a 3D wave equation. The goal is to simulate wave propagation in a 3D medium using the finite difference method and visualize the results.
This repository builds on the work done by Guilliaume Carraux.

## Optimizations

These optimisations are suggestions made by Guilliaume for the forward simulation:

---

=> these will provide a clear performance boost:

- place model in cuda constant memory (- find a way to compress information for 3d model). The model lookup for K a each step doubles simulation time.
- use less kernels (more complex ones) for adding RK4 values together.

=> worth trying:

- tweak block size
- preload some values in kernel calls (ex K and Rho a gcoords and around => will lookup once only)

=> to find still:

- analyze redundant computations
