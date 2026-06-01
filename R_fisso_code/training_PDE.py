import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from blind_model import N
import copy
import matplotlib.pyplot as plt
from no_gate_model import Basis_Unit_no_gate
from bu_gate_model import modello_intermedio

def calcola_laplaciano(psi: torch.Tensor, r_batch: torch.Tensor) -> torch.Tensor:
    """
    Calcola l'operatore Laplaciano (somma delle derivate seconde spaziali) 
    della funzione d'onda rispetto alle coordinate spaziali.
    """
    gradiente_psi = torch.autograd.grad(
        outputs=psi, 
        inputs=r_batch, 
        grad_outputs=torch.ones_like(psi), 
        create_graph=True,
        retain_graph=True
    )[0] 
    
    # derivata seconda: d2/dx2 + d2/dy2 + d2/dz2
    laplaciano_psi = torch.zeros_like(psi)
    
    for i in range(3): 
        derivata_seconda = torch.autograd.grad(
            outputs=gradiente_psi[:, i], 
            inputs=r_batch, 
            grad_outputs=torch.ones_like(gradiente_psi[:, i]), 
            create_graph=True,
            retain_graph=True   
        )[0]
        laplaciano_psi += derivata_seconda[:, i]
        
    return laplaciano_psi


def calcola_potenziale(r_batch: torch.Tensor, R_fisso: float, epsilon: float = 0.05) -> torch.Tensor:
    """
    Calcola il potenziale Coulombiano per la molecola H2+.
    """
    # Recupero il device (CPU o GPU) su cui si trovano i dati
    dispositivo = r_batch.device 
    
    # Creo i nuclei direttamente sul device corretto
    nucleo_sx = torch.tensor([-R_fisso, 0.0, 0.0], device=dispositivo)
    nucleo_dx = torch.tensor([R_fisso, 0.0, 0.0], device=dispositivo)
    
    dist_nucleo_sx = torch.norm(r_batch - nucleo_sx, dim=-1)
    dist_nucleo_dx = torch.norm(r_batch - nucleo_dx, dim=-1)
    
    V = -1.0 / torch.sqrt(dist_nucleo_sx**2 + epsilon**2) - 1.0 / torch.sqrt(dist_nucleo_dx**2 + epsilon**2)
    return V


def calcola_Loss_PDE(psi: torch.Tensor, r_batch: torch.Tensor, E_corrente: torch.Tensor, R_fisso: float) -> torch.Tensor:
    """
    Assembla l'equazione di Schrödinger per calcolare il residuo e la Loss PDE.
    """
    laplaciano = calcola_laplaciano(psi, r_batch)
    V = calcola_potenziale(r_batch, R_fisso)
    
    H_psi = -0.5 * laplaciano + V * psi
    
    residuo = H_psi - (E_corrente * psi)
    
    # Ritorna il Mean Squared Error del residuo
    return torch.mean(residuo**2)

if __name__ == '__main__':
    print("Seleziona il modello da addestrare:", '\n',
          "1. Modello con Psi_tot = NN completamente blind (senza bias, energia parametro)", '\n',
          "2. Modello con Psi_tot = Psi_LCAO + N (energia: parametro)", '\n',
        "3. Modello con Psi_tot = Psi_LCAO + f ° N (energia: parametro)")

    scelta = input("Inserisci 1, 2 o 3: ")
    if scelta == '1':
        print("Hai scelto il modello completamente blind (senza bias, energia: parametro).")
        model = N(input_dim=3, n_hidden=2, n_neurons=16)
    elif scelta == '2':
        print("Hai scelto il modello Psi_tot = Psi_LCAO + N  (energia: parametro)(no gate).")
        model = Basis_Unit_no_gate(input_dim=3, n_hidden=2, n_neurons=16)
    elif scelta == '3':
        print("Hai scelto il modello con Psi_tot = Psi_LCAO + f ° N (energia: parametro).")
        model = modello_intermedio(input_dim=3, n_hidden=2, n_neurons=16)  

    # R fissata
    R_fisso = 1 

    ottimizzatore = optim.Adam(model.parameters(), lr=8*10**(-3))

    epoche = 1500
    punti_per_epoca = 5000
    valori_loss = []          
    miglior_loss = float('inf')  
    migliori_pesi = None
    valori_energia = []

    for epoca in range(epoche):
        
        r_batch = (torch.rand(punti_per_epoca, 3) * 20 ) - 10 # distribuz. tra -10 e +10 a.u.
        r_batch.requires_grad_(True) # per calcolare le derivate

        # forward pass
        psi = model(r_batch, R_fisso)

        loss_PDE = calcola_Loss_PDE(psi, r_batch, model.E, R_fisso)

        # backpropagation 
        loss_PDE.backward() 
        ottimizzatore.step()
        ottimizzatore.zero_grad()

        # salvataggio dati (loss ed energia)
        loss_attuale = loss_PDE.item()
        valori_loss.append(loss_attuale)
        valori_energia.append(model.E.item())

        if loss_attuale < miglior_loss:
            miglior_loss = loss_attuale
            migliori_pesi = copy.deepcopy(model.state_dict())
            if epoca % 400 == 0:
                print(f"Epoca {epoca} | Loss PDE: {miglior_loss:.6f} | Energia E: {model.E.item():.4f}")

    print(f"Training completato. La loss minima raggiunta è stata: {miglior_loss:.6f}")
    model.load_state_dict(migliori_pesi)

    energia_finale = model.E.item()
    print(f"L'Energia calcolata dalla rete per R=1.0 è: {energia_finale:.6f} a.u.")


   # disegno della loss function
    plt.plot(valori_loss, color='blue', linewidth=2)
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.title('Andamento della Loss')
    plt.yscale('log')
    plt.show()

    # disegno dell'andamento dei valori di energia 
    plt.figure(figsize=(8, 5))
    plt.plot(valori_energia, color='green', linewidth=2, label='Energia stimata dal PINN')
    plt.xlabel('Epoca')
    plt.ylabel('Energia (a.u.)')
    plt.grid(True, alpha=0.3)
    plt.title('Convergenza dell\'Energia')
    plt.legend()
    plt.show()

    # Disegno della funzione d'onda 
    asse_x = torch.linspace(-5, 5, 1000)
    punti_plot = torch.zeros(1000, 3)
    punti_plot[:, 0] = asse_x

    # Valuto il modello
    # Occorre usare torch.no_grad() perché per fare un grafico non servono le derivate
    # Questo rende il calcolo istantaneo e fa risparmiare memoria
    with torch.no_grad():
        psi_valori = model(punti_plot, R_fisso)
        psi_valori_norm = psi_valori / torch.max(psi_valori)

        if hasattr(model, 'calcola_LCAO'): 
            psi_LCAO_valori = model.calcola_LCAO(punti_plot, R_fisso)
            psi_LCAO_norm = psi_LCAO_valori / torch.max(psi_LCAO_valori)
        else:
            psi_LCAO_valori = None
            psi_LCAO_norm = None

    plt.figure(figsize=(10, 6))

    plt.plot(asse_x.numpy(), psi_valori_norm.numpy(), color='purple', linewidth=2.5, label='Psi Totale (output modello)')
  
    if psi_LCAO_valori is not None:
        plt.plot(asse_x.numpy(), psi_LCAO_norm.numpy(), color='blue', linestyle='--', linewidth=2, label='Psi LCAO classica (Base)')
        plt.title("Confronto: LCAO classica vs. Correzione PINN lungo l'asse internucleare")
    else:
        # Titolo alternativo se stiamo usando la rete cieca (Modello 1)
        plt.title("Funzione d'onda appresa dalla rete neurale pura (Blind model)")

    # Aggiungo i due nuclei 
    plt.scatter([-R_fisso, R_fisso], [0, 0], color='red', s=60, zorder=5, label='Nuclei')
    plt.axvline(x=-R_fisso, color='red', linestyle=':', alpha=0.5)
    plt.axvline(x=R_fisso, color='red', linestyle=':', alpha=0.5)

    # Dettagli del grafico
    plt.xlabel('x [a.u.]')
    plt.ylabel('Ampiezza Psi')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.show()