import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42) 

class Basis_Unit_no_gate(nn.Module):
    def __init__(self,input_dim,  n_hidden, n_neurons):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        # L'Energia (per ora) è un parametro scalare addestrabile (lo inizializzo a -1.0)
        self.E = nn.Parameter(torch.tensor([-1.0]))
        
        # Creazione della rete neurale (Basis Unit)
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
        #layers.append(nn.Sigmoid())  senza sigmoide per la correzione

        self.model = nn.Sequential(*layers)
    
    # metodi fisici
    def calcola_LCAO(self, r, R):
        dispositivo = r.device 
        nucleo_sx = torch.tensor([-R, 0.0, 0.0], device=dispositivo)
        nucleo_dx = torch.tensor([R, 0.0, 0.0], device=dispositivo)

        phi_sx = torch.exp(-torch.norm(r - nucleo_sx, dim=-1))
        phi_dx = torch.exp(-torch.norm(r - nucleo_dx, dim=-1))

        return phi_sx + phi_dx
    
    def funzione_f_gate (self, r):
        # Per ora è una funzione identità, ma in futuro potrei renderla più complessa
        return 1
    
    def forward(self, X, R):
        """
        X: coordinate spaziali (N, 3)
        R: distanza interatomica (scalare, per ora)
        """
        psi_LCAO = self.calcola_LCAO(X, R)

        funzione_f_gate = self.funzione_f_gate(X)

        correzione_BU = self.model(X).squeeze(-1) 

        psi_tot = psi_LCAO + funzione_f_gate * correzione_BU

        return psi_tot