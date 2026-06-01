import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42) # Così sto bloccando i valori dei "pesi" e i "bias" (parametri addestrabili) con cui vengono inizializzati 
# È fondamentale per la ricerca scientifica. Se modifico il numero di neuroni e vedo che la rete migliora, 
# devo essere sicuro che sia merito della mia modifica e non di un'inizializzazione casuale fortunata

class N(nn.Module):
    def __init__(self,input_dim,  n_hidden, n_neurons):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        # L'Energia (per ora) è un parametro scalare addestrabile (lo inizializzo a -1.0)
        self.E = nn.Parameter(torch.tensor([-1.0]))

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
    
    def forward(self, X, R):
        return self.model(X).squeeze(-1)
    
    def compute_loss(self, y_pred, y_vero):
        return torch.mean((y_vero- y_pred) ** 2) # mean square error
    
# test fittizio 
if __name__ == '__main__':
    # Creo la rete con i parametri del paper per la Basis Unit
    # input_dim = 3 (x, y, z), n_hidden = 2 layer, n_neurons = 16
    modello = N(input_dim=3, n_hidden=2, n_neurons=16)
    print("Architettura della rete:")
    print(modello, "\n")

    X_batch = torch.rand(5, 3) # 5 punti casuali
    Y_target = torch.tensor([0.5, -0.2, 0.1, 0.9, -0.5])

    previsioni = modello(X_batch)
    print(f"Le 5 previsioni iniziali (casuali) della rete: {previsioni.detach().numpy()}")
    print(f"I 5 valori che dovrebbe indovinare:          {Y_target.numpy()}")

    loss = modello.compute_loss(previsioni, Y_target)
    print(f"\nLoss calcolata (Errore quadratico medio): {loss.item():.4f}")