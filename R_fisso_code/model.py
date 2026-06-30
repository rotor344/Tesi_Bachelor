import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42) 

# Basis Unit
class Basis_Unit(nn.Module):
    def __init__(self,input_dim,  n_hidden, n_neurons):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        # In this case the energy is a trainable scalar parameter (initialized to -1.0)
        self.E = nn.Parameter(torch.tensor([-1.0]))
        
        # Creation of the neural network (Basis Unit)
        layers = []

        # input
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())

        # inner 
        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        # output
        layers.append(nn.Linear(n_neurons, 1)) 
        # without sigmoid for correction

        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X).squeeze(-1) 
    
# Gate
class Gate(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons

        layers = []
        # input
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())

        # inner 
        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        # output
        layers.append(nn.Linear(n_neurons, 1)) 
        # Non essendo un hidden layer, non metto la sigmoide: voglio che il gate possa essere anche negativo o maggiore di 1, per modulare la correzione in modo più flessibile

        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X).squeeze(-1)

# Intermediate model with Basis Unit and Gate, made symmetric
class modello_sym(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        self.basis_unit = Basis_Unit(input_dim, n_hidden, n_neurons)
        self.gate = Gate(input_dim=1, n_hidden=0, n_neurons=10)  # the gate has its own architecture    

        # Sharing the same energy parameter between the Basis Unit and the intermediate model
        self.E = self.basis_unit.E  
        # Parity operator (+1 : symmetric ground state, -1 : antisymmetric first excited state)
        self.P = 1.0

    # fisica
    def calcola_LCAO(self, r, R):
        dispositivo = r.device 
        nucleo_sx = torch.tensor([-R, 0.0, 0.0], device=dispositivo)
        nucleo_dx = torch.tensor([R, 0.0, 0.0], device=dispositivo)

        phi_sx = torch.exp(-torch.norm(r - nucleo_sx, dim=-1))
        phi_dx = torch.exp(-torch.norm(r - nucleo_dx, dim=-1))

        return phi_sx + phi_dx
    
    
    def forward(self, X, R):

        psi_LCAO = self.calcola_LCAO(X, R)

        R_tensor = torch.tensor([R], dtype=torch.float32, device=X.device)
        gate_value = self.gate(R_tensor) # The gate value is a scalar that modulates the neural correction

        # neural correction in +X
        correzione_pos = self.basis_unit(X)
        N_pos = gate_value * correzione_pos

        # neural correction in -X (symmetric)
        correzione_neg = self.basis_unit(-X)
        N_neg = gate_value * correzione_neg

        N_simmetrica = 0.5 * (N_pos + self.P * N_neg)
        
        return psi_LCAO +  N_simmetrica 
    
 
    
