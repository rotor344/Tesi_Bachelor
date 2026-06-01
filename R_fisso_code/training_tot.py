import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from blind_model import N
import copy
import matplotlib.pyplot as plt
from no_gate_model import Basis_Unit_no_gate
from bu_gate_model import modello_intermedio
from model import modello_sym

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


def calcola_potenziale(r_batch: torch.Tensor, R_fisso: float, epsilon: float = 0.005) -> torch.Tensor:
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

def calcola_Loss_BC(psi: torch.Tensor, r_batch: torch.Tensor, R_fisso: float) -> torch.Tensor:
    """
    Calcola la Loss per le condizioni al contorno (psi deve essere 0 a distanza infinita).
    """
    dispositivo = r_batch.device
    nucleo_sx = torch.tensor([-R_fisso, 0.0, 0.0], device=dispositivo)
    nucleo_dx = torch.tensor([R_fisso, 0.0, 0.0], device=dispositivo)

    dist_nucleo_sx = torch.norm(r_batch - nucleo_sx, dim=-1)
    dist_nucleo_dx = torch.norm(r_batch - nucleo_dx, dim=-1)

    # Identifico i punti "lontani"
    soglia_distanza = 9.5
    punti_lontani = (dist_nucleo_sx > soglia_distanza) & (dist_nucleo_dx > soglia_distanza)

    if torch.sum(punti_lontani) == 0:
        return torch.tensor(0.0)  # Nessun punto lontano, nessuna penalizzazione

    psi_lontano = psi[punti_lontani]
    
    # Penalizzo se psi è troppo grande nei punti lontani
    return torch.mean(psi_lontano**2)
    

if __name__ == '__main__':
    print("Seleziona il modello da addestrare:", '\n',
        "1. Modello con Psi_tot = NN completamente blind (senza bias, energia: parametro)", '\n',
        "2. Modello con Psi_tot = Psi_LCAO + N (energia: parametro)", '\n',
        "3. Modello con Psi_tot = Psi_LCAO + f ° N (energia: parametro)", '\n',
        "4. Modello con Psi_tot = Psi_LCAO + f ° N (energia: parametro) e simmetria esplicita")

    scelta = input("Digita 1, 2, 3 o 4: ")
    if scelta == '1':
        print("Hai scelto il modello completamente blind (senza bias, energia: parametro).")
        model = N(input_dim=3, n_hidden=2, n_neurons=16)
    elif scelta == '2':
        print("Hai scelto il modello Psi_tot = Psi_LCAO + N  (energia: parametro)(no gate).")
        model = Basis_Unit_no_gate(input_dim=3, n_hidden=2, n_neurons=16)
    elif scelta == '3':
        print("Hai scelto il modello con Psi_tot = Psi_LCAO + f ° N (energia: parametro).")
        model = modello_intermedio(input_dim=3, n_hidden=2, n_neurons=16)  
    elif scelta == '4':
        print("Hai scelto il modello con Psi_tot = Psi_LCAO + f ° N (energia: parametro) e simmetria esplicita.")
        model = modello_sym(input_dim=3, n_hidden=2, n_neurons=16)  

    # R fissata
    R_fisso = 1 

    ottimizzatore = optim.Adam(model.parameters(), lr=8*10**(-3))

    epoche = 2000
    punti_per_epoca = 6000
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
        loss_BC = calcola_Loss_BC(psi, r_batch, R_fisso)
        loss_totale = loss_PDE + loss_BC  

        # backpropagation 
        loss_totale.backward() 
        ottimizzatore.step()
        ottimizzatore.zero_grad()

        # salvataggio dati (loss ed energia)
        loss_attuale = loss_totale.item()
        valori_loss.append(loss_attuale)
        valori_energia.append(model.E.item())

        if loss_attuale < miglior_loss:
            miglior_loss = loss_attuale
            migliori_pesi = copy.deepcopy(model.state_dict())
            if epoca % 400 == 0:
                print(f"Epoca {epoca} | Loss: {miglior_loss:.6f} | Energia E: {model.E.item():.4f}")

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

    # Disegno della funzione d'onda [Normalizzazione sbagliata]
    asse_x = torch.linspace(-5, 5, 1000)
    punti_plot = torch.zeros(1000, 3)
    punti_plot[:, 0] = asse_x

    # Valuto il modello
    # Occorre usare torch.no_grad() perché per fare un grafico non servono le derivate
    # Questo rende il calcolo istantaneo e fa risparmiare memoria
    with torch.no_grad():
        psi_valori = model(punti_plot, R_fisso)
        psi_valori_norm = psi_valori/ torch.max(psi_valori)

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

  # ---------------------------------------------------------
    # NORMALIZZAZIONE Fisicamente corretta: integrazione numerica della funzione d'onda al quadrato 
  
    # Calcolo della costante di normalizzazione L^2 in 3D (integrale numerico su una griglia 3D)
    # Metodo: Somma di Riemann

    box_size = 10.0
    n_grid_3d = 60 # 60x60x60 = 216.000 punti per stimare il volume 3D
    asse_3d = torch.linspace(-box_size, box_size, n_grid_3d)
    xg, yg, zg = torch.meshgrid(asse_3d, asse_3d, asse_3d, indexing='ij')
    
    # Appiattire la griglia per passarla al modello
    punti_3d = torch.stack([xg.flatten(), yg.flatten(), zg.flatten()], dim=-1)
    
    # Elemento di volume dV = dx * dy * dz
    dx_3d = (2.0 * box_size) / (n_grid_3d - 1)
    dV = dx_3d ** 3

    with torch.no_grad():
        psi_3d_valori = model(punti_3d, R_fisso)
        # Integrazione 3D numerica (Somma di Riemann sul volume)
        volume_pinn = torch.sum(psi_3d_valori**2) * dV
        norma_pinn_3D = 1.0 / torch.sqrt(volume_pinn)

        if hasattr(model, 'calcola_LCAO'): 
            psi_LCAO_3d = model.calcola_LCAO(punti_3d, R_fisso)
            volume_lcao = torch.sum(psi_LCAO_3d**2) * dV
            norma_lcao_3D = 1.0 / torch.sqrt(volume_lcao)
        else:
            norma_lcao_3D = None

    # Disegno del grafico 1D con la funzione d'onda normalizzata secondo la vera normalizzazione fisica 3D
    asse_x_1d = torch.linspace(-box_size, box_size, 1000)
    punti_plot_1d = torch.zeros(1000, 3)
    punti_plot_1d[:, 0] = asse_x_1d

    with torch.no_grad():
        psi_1d_valori = model(punti_plot_1d, R_fisso)
        
        # Moltiplico per la costante 3D corretta
        psi_valori_norm = psi_1d_valori * norma_pinn_3D

        if norma_lcao_3D is not None:
            psi_LCAO_1d_valori = model.calcola_LCAO(punti_plot_1d, R_fisso)
            psi_LCAO_norm = psi_LCAO_1d_valori * norma_lcao_3D

    plt.figure(figsize=(10, 6))

    plt.plot(asse_x_1d.numpy(), psi_valori_norm.numpy(), color='purple', linewidth=2.5, label='Psi Totale (Norm. L^2 3D)')
  
    if norma_lcao_3D is not None:
        plt.plot(asse_x_1d.numpy(), psi_LCAO_norm.numpy(), color='blue', linestyle='--', linewidth=2, label='Psi LCAO classica (Norm. L^2 3D)')
        plt.title("Confronto: LCAO vs PINN (Vera Normalizzazione Fisica 3D)")
    else:
        plt.title("Funzione d'onda appresa dalla rete (Vera Normalizzazione Fisica 3D)")

    # Aggiungo i due nuclei 
    plt.scatter([-R_fisso, R_fisso], [0, 0], color='red', s=60, zorder=5, label='Nuclei')
    plt.axvline(x=-R_fisso, color='red', linestyle=':', alpha=0.5)
    plt.axvline(x=R_fisso, color='red', linestyle=':', alpha=0.5)

    # Dettagli del grafico
    plt.xlabel('x [a.u.]')
    plt.ylabel('Ampiezza Psi Normalizzata')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.show()