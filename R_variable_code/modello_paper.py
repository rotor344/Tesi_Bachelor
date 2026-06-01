import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42) 

class Basis_Unit(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())

        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        layers.append(nn.Linear(n_neurons, 1)) 
        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X).squeeze(-1) 
    
class Gate(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())

        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        layers.append(nn.Linear(n_neurons, 1)) 
        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X).squeeze(-1)

class Energy_Unit(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())

        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        self.out_layer = nn.Linear(n_neurons, 1)
        # Inizializzo il bias dell'energia a -1 (come nel paper) 
        # per aiutare la rete a partire vicina al target (-0.5 / -1.8)
        nn.init.constant_(self.out_layer.bias, -1.0)
        layers.append(self.out_layer) 
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X).squeeze(-1) 

# architettura del paper tradotta fedelmente
class modello_paper(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        # La basis unit prende in input le 2 orbite (phi_1, phi_2)
        self.basis_unit = Basis_Unit(input_dim=2, n_hidden=n_hidden, n_neurons=n_neurons)
        
        # netDecay (Gate) e Energy Unit
        self.gate = Gate(input_dim=1, n_hidden=0, n_neurons=10)  
        self.energy_unit = Energy_Unit(input_dim=1, n_hidden=1, n_neurons=32)  

        self.P = 1.0 # Simmetria di inversione (P=1 per lo stato fondamentale)

    def calcola_phi(self, r, R_batch):
        nucleo_sx = torch.zeros_like(r)
        nucleo_dx = torch.zeros_like(r)
        nucleo_sx[:, 0] = -R_batch.squeeze()
        nucleo_dx[:, 0] = R_batch.squeeze()

        r1 = torch.norm(r - nucleo_sx, dim=-1)
        r2 = torch.norm(r - nucleo_dx, dim=-1)

        # Attivazione atomica 
        phi_sx = torch.exp(-r1)
        phi_dx = torch.exp(-r2)

        return phi_sx, phi_dx
    
    def forward(self, X, R_batch):
        # Basi atomiche
        phi_sx, phi_dx = self.calcola_phi(X, R_batch)
        psi_LCAO = phi_sx + self.P * phi_dx

        # Features per il punto X [batch, 2]
        features_pos = torch.stack([phi_sx, phi_dx], dim=-1)
        
        # Features per il punto opposto -X
        # scambiare phi_sx e phi_dx è fisicamente identico a calcolare la norma su -X
        features_neg = torch.stack([phi_dx, phi_sx], dim=-1)

        # Reti su R
        gate_value = self.gate(R_batch).squeeze(-1) 
        energy = self.energy_unit(R_batch).squeeze(-1)

        # correzione simmetrica (Basis Unit)
        B_pos = self.basis_unit(features_pos)
        B_neg = self.basis_unit(features_neg)
        NN_simmetrico = B_pos + self.P * B_neg
        
        # Ansatz LCAO + Correzione
        correzione_totale = gate_value * NN_simmetrico
        psi_tot = psi_LCAO + correzione_totale
        
        return psi_tot, energy