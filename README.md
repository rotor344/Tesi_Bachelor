## Bachelor's Thesis Project: BSc in Physics, Unimib

The electronic Schrödinger equation is the heart of quantum chemistry, but its exact solution is impossible for systems with more than one electron and complex even for the simplest molecules. 

This Bachelor's Thesis project explores an innovative paradigm: **Physics-Informed Neural Networks (PINNs)**. Instead of solving the equation through complex variational methods or traditional numerical grids (e.g., LCAO), I train a neural network to "understand" the physics of the system. 

Specifically, I built a model for the $H_2^+$ molecular ion that:
1. Employs a hybrid Ansatz, combining classical physical knowledge (LCAO) with the flexibility of a deep neural network.
2. Is trained to directly minimize the residual of the differential equation (PDE Loss) and strictly respect the boundary conditions (BC Loss).
3. Is capable of autonomously extracting both the **dissociation curve $E(R)$** and the **interatomic forces** (via automatic differentiation, `autograd`).

The results demonstrate how the integration of physical constraints (such as inversion symmetry, spatial gates, and potential cut-offs) is crucial to properly guide the network towards the exact solution while avoiding explosive singularities.