import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)

class Basis_Unit(nn.Module):
    def __init__(self,input_dim,  n_hidden, n_neurons):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        
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
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons

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
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons

        layers = []
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())

        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        self.out_layer = nn.Linear(n_neurons, 1)
        # Inizializzo il bias dell'energia a -1 (come nel paper) 
        nn.init.constant_(self.out_layer.bias, -1.0)
        layers.append(self.out_layer) 
        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X).squeeze(-1) 


class modello_v0(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        self.basis_unit = Basis_Unit(input_dim=input_dim, n_hidden=n_hidden, n_neurons=n_neurons)
        
        self.gate = Gate(input_dim=1, n_hidden=0, n_neurons=10) 
        self.energy_unit = Energy_Unit(input_dim=1, n_hidden=1, n_neurons=32)  
        
        self.P = 1.0 # Parità per stato simmetrico
        print(f"[INFO] Inizializzato modello con input_dim={input_dim} per la Basis_Unit")

    def calcola_LCAO(self, r, R_batch):
        nucleo_sx = torch.zeros_like(r)
        nucleo_dx = torch.zeros_like(r)
        nucleo_sx[:, 0] = -R_batch.squeeze()
        nucleo_dx[:, 0] = R_batch.squeeze()

        r1 = torch.norm(r - nucleo_sx, dim=-1)
        r2 = torch.norm(r - nucleo_dx, dim=-1)

        phi_sx = torch.exp(-r1)
        phi_dx = torch.exp(-r2)

        return phi_sx + phi_dx
    
    def forward(self, X, R_batch):
        # Base Fisica
        psi_LCAO = self.calcola_LCAO(X, R_batch)

        gate_value = self.gate(R_batch).squeeze(-1) 
        energy = self.energy_unit(R_batch).squeeze(-1) 

        # Correzione spaziale (in X e -X per garantire la simmetria)
        correzione_pos = self.basis_unit(X)
        correzione_neg = self.basis_unit(-X)
        N_simmetrica = 0.5 * (correzione_pos + self.P * correzione_neg)
        
        correzione_totale = gate_value * N_simmetrica 
        
        psi_tot = psi_LCAO + correzione_totale
        
        return psi_tot, energy
    

# MODELLO CON INPUT_DIM=4 (x, y, z, R) PER LA BASIS UNIT, INVECE DI 3 SOLO PER LE COORDINATE SPAZIALI. 
# IL GATE E L'ENERGY UNIT RESTANO CON INPUT_DIM=1 (R)

# Basis Unit
class Basis_Unit2(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())

        # Hidden layers
        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        # Output layer
        layers.append(nn.Linear(n_neurons, 1)) 

        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X).squeeze(-1) 
    
# Gate
class Gate2(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons

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

# Energy Unit 
class Energy_Unit2(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons

        layers = []
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())

        for _ in range(1, n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        self.out_layer = nn.Linear(n_neurons, 1)
        # Inizializzo il bias dell'energia a -1 (come nel paper)
        nn.init.constant_(self.out_layer.bias, -1.0)
        layers.append(self.out_layer) 
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X).squeeze(-1) 


# Modello v0 modificato a 4 Input (x, y, z, R)
class modello_v2(nn.Module):
    def __init__(self, input_dim, n_hidden, n_neurons):
        super().__init__()
        self.basis_unit = Basis_Unit2(input_dim=input_dim, n_hidden=n_hidden, n_neurons=n_neurons)
        self.gate = Gate2(input_dim=1, n_hidden=0, n_neurons=10) 
        self.energy_unit = Energy_Unit2(input_dim=1, n_hidden=1, n_neurons=32)  
        
        self.P = 1.0 

        print(f"[INFO] Inizializzato modello con input_dim={input_dim} per la Basis_Unit")

    def calcola_LCAO(self, r, R_batch):
        nucleo_sx = torch.zeros_like(r)
        nucleo_dx = torch.zeros_like(r)
        nucleo_sx[:, 0] = -R_batch.squeeze()
        nucleo_dx[:, 0] = R_batch.squeeze()

        r1 = torch.norm(r - nucleo_sx, dim=-1)
        r2 = torch.norm(r - nucleo_dx, dim=-1)

        phi_sx = torch.exp(-r1)
        phi_dx = torch.exp(-r2)

        return phi_sx + phi_dx
    
    def forward(self, X, R_batch):
   
        psi_LCAO = self.calcola_LCAO(X, R_batch)
        gate_value = self.gate(R_batch).squeeze(-1) 
        energy = self.energy_unit(R_batch).squeeze(-1) 

        # X concatenated with R_batch to form a 4D input for the Basis Unit 
        input_pos = torch.cat([X, R_batch], dim=-1)
        
        # For the geometric parity inversion, the spatial coordinates are inverted (-X), 
        # while the parameter R remains unchanged.
        input_neg = torch.cat([-X, R_batch], dim=-1)

        correzione_pos = self.basis_unit(input_pos)
        correzione_neg = self.basis_unit(input_neg)
        N_simmetrica = 0.5 * (correzione_pos + self.P * correzione_neg)
        
        # Total ansatz
        correzione_totale = gate_value * N_simmetrica 
        psi_tot = psi_LCAO + correzione_totale
        
        return psi_tot, energy