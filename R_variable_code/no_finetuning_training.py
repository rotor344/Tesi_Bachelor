import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from model_v0 import modello_v0

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


def calcola_potenziale(r_batch: torch.Tensor, R_batch: float, epsilon: float = 0.005) -> torch.Tensor:
    """
    Calcola il potenziale Coulombiano per la molecola H2+.
    """
    # Recupero il device (CPU o GPU) su cui si trovano i dati
    dispositivo = r_batch.device 
    
    nucleo_sx = torch.zeros_like(r_batch)
    nucleo_dx = torch.zeros_like(r_batch)
    # Assegno -R e +R alla colonna delle X (indice 0)
    # squeeze() per togliere la dimensione extra [batch, 1] -> [batch]
    nucleo_sx[:, 0] = -R_batch.squeeze()
    nucleo_dx[:, 0] = R_batch.squeeze()
    
    dist_nucleo_sx = torch.norm(r_batch - nucleo_sx, dim=-1)
    dist_nucleo_dx = torch.norm(r_batch - nucleo_dx, dim=-1)
    
    V = -1.0 / torch.sqrt(dist_nucleo_sx**2 + epsilon**2) - 1.0 / torch.sqrt(dist_nucleo_dx**2 + epsilon**2)
    return V


def calcola_Loss_PDE(psi: torch.Tensor, r_batch: torch.Tensor, E_corrente: torch.Tensor, R_batch: float) -> torch.Tensor:
    """
    Assembla l'equazione di Schrödinger per calcolare il residuo e la Loss PDE.
    """
    laplaciano = calcola_laplaciano(psi, r_batch)
    V = calcola_potenziale(r_batch, R_batch)
    
    H_psi = -0.5 * laplaciano + V * psi
    
    residuo = H_psi - (E_corrente * psi)
    
    # Ritorna il Mean Squared Error del residuo
    return torch.mean(residuo**2)

def calcola_Loss_BC(psi: torch.Tensor, r_batch: torch.Tensor, R_batch: float) -> torch.Tensor:
    """
    Calcola la Loss per le condizioni al contorno (psi deve essere 0 a distanza infinita).
    """
    dispositivo = r_batch.device
    nucleo_sx = torch.zeros_like(r_batch)
    nucleo_dx = torch.zeros_like(r_batch)
    # Assegno -R e +R alla colonna delle X (indice 0)
    # squeeze() per togliere la dimensione extra [batch, 1] -> [batch]
    nucleo_sx[:, 0] = -R_batch.squeeze()
    nucleo_dx[:, 0] = R_batch.squeeze()

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
        "1. Modello v0 (n_layers e n_neurons come paper)", '\n',
        "2. Modello con ...", '\n',
        "3. Modello con ...", '\n',
        "4. Modello con ...")

    scelta = input("Digita 1, 2, 3 o 4: ")
    if scelta == '1':
        print("Hai scelto il modello v_0 (come paper).")
        model = modello_v0(input_dim=3, n_hidden=2, n_neurons=16)
    elif scelta == '2':
        print("Hai scelto il modello ..")
        model = 0
    elif scelta == '3':
        print("Hai scelto il modello ..")
        model = 0 
    elif scelta == '4':
        print("Hai scelto il modello ..")
        model = 0


    ottimizzatore = optim.Adam(model.parameters(), lr=8*10**(-3))

    epoche = 1500
    punti_per_epoca = 5000
    valori_loss = []          
    miglior_loss = float('inf')  
    migliori_pesi = None
    valori_energia = []

    valori_loss_PDE = []
    valori_loss_BC = []

    for epoca in range(epoche):
        
        r_batch = (torch.rand(punti_per_epoca, 3) * 20 ) - 10 # distribuz. tra -10 e +10 a.u.
        r_batch.requires_grad_(True) # per calcolare le derivate

        R_batch = (3.0 - 0.2) * torch.rand(punti_per_epoca, 1) + 0.2

        # forward pass
        psi, energy = model(r_batch, R_batch)

        loss_PDE = calcola_Loss_PDE(psi, r_batch, energy, R_batch)
        loss_BC = calcola_Loss_BC(psi, r_batch, R_batch)
        loss_totale = loss_PDE + loss_BC  

        # backpropagation 
        loss_totale.backward() 
        ottimizzatore.step()
        ottimizzatore.zero_grad()

        # salvataggio dati (loss ed energia)
        loss_attuale_PDE = loss_PDE.item()
        valori_loss_PDE.append(loss_attuale_PDE) 
        loss_attuale_BC = loss_BC.item()
        valori_loss_BC.append(loss_attuale_BC)

        # salvataggio dati (loss ed energia)
        loss_attuale = loss_totale.item()
        valori_loss.append(loss_attuale)
        valori_energia.append(torch.mean(energy).item())

        if loss_attuale < miglior_loss:
            miglior_loss = loss_attuale
            migliori_pesi = copy.deepcopy(model.state_dict())
            if epoca % 400 == 0:
                print(f"Epoca {epoca} | Loss: {miglior_loss:.6f} | Energia E: {torch.mean(energy).item():.4f}")

    print(f"Training completato. La loss minima raggiunta è stata: {miglior_loss:.6f}")
    model.load_state_dict(migliori_pesi)

    energia_finale = torch.mean(energy).item()
    print(f"L'Energia calcolata dalla rete per ... è: {energia_finale:.6f} a.u.")

    # disegno delle Loss
    plt.figure(figsize=(10, 6))
    
    plt.plot(valori_loss_PDE, color='red', linewidth=2, label='Loss PDE')
    plt.plot(valori_loss_BC, color='green', linewidth=2, label='Loss BC')
    
    plt.plot(valori_loss, color='blue', linewidth=4, alpha=0.3, label='Total Loss')
    
    plt.xlabel('Epoca')
    plt.ylabel('Valore Loss')
    plt.grid(True, alpha=0.3)
    plt.title('Andamento delle componenti della Loss')
    plt.yscale('log') 
    plt.legend()
    plt.show()

    # disegno della curva di dissociazione E(R) 
    # Creo un tensore con 200 distanze R ordinate, da 0.2 a 3.0
    # unsqueeze(1) per farlo diventare un vettore colonna [200, 1] compatibile con la rete
    R_test = torch.linspace(0.2, 3.5, 200).unsqueeze(1)
    
    # Chiedo all'Energy Unit di calcolare i valori per queste distanze
    # torch.no_grad() perché sto solo valutando il modello finito
    with torch.no_grad():
        E_el = model.energy_unit(R_test).squeeze(-1).numpy()
        
    R_np = R_test.squeeze().numpy()
    distanza_internucleare = 2.0 * R_np
    # Calcolo la repulsione nucleare (1 / Distanza Totale)
    repulsione = 1.0 / distanza_internucleare

    # Energia totale = Energia elettronica + Repulsione nucleare
    E_tot = E_el + repulsione

    plt.figure(figsize=(8, 5))
    plt.plot(distanza_internucleare, E_tot, color='green', linewidth=2.5, label='Neural E(R)+ Repulsione')
    
    # valori di riferimento esatti ( H. Wind, 1965)
    # R da 0.2 a 4.0 con passo di 0.1
    R_exact = np.round(np.arange(0.2, 4.1, 0.1), 2)  
    
    # REFERENCE: Valori dell'Energia Elettronica Esatta estratti dal paper
    E_elec_exact = np.array([
        -1.8008, -1.6715, -1.5545, -1.4518, -1.3623, -1.2843, -1.2159, -1.1558,
        -1.1026, -1.0554, -1.0132, -0.9754, -0.9415, -0.9109, -0.8832, -0.8582, 
        -0.8355, -0.8149, -0.7961, -0.7790, -0.7634, -0.7492, -0.7363, -0.7244, 
        -0.7136, -0.7037, -0.6946, -0.6863, -0.6786, -0.6716, -0.6651, -0.6591, 
        -0.6536, -0.6485, -0.6437, -0.6392, -0.6351, -0.6312, -0.6276
    ])
    
    # Calcolo della Distanza Internucleare e dell'Energia Totale Esatta
    distanza_internucleare_exact = 2.0 * R_exact
    repulsione_exact = 1.0 / distanza_internucleare_exact
    E_tot_exact = E_elec_exact + repulsione_exact

    #plt.plot(distanza_internucleare_exact, E_elec_exact, color='black', marker='o', fillstyle='none', linestyle='none', markersize=6, label='Energia Elettronica (Esatta)')
    
    plt.plot(distanza_internucleare_exact, E_tot_exact, color='red', marker='o', linestyle='none', markersize=6, label='Reference')

    # LCAO 
    D = distanza_internucleare
    
    # Formule analitiche degli integrali per H2+ (orbitali 1s)
    S = np.exp(-D) * (1.0 + D + (D**2) / 3.0)                       # Integrale di Sovrapposizione
    J = (1.0 / D) - np.exp(-2.0 * D) * (1.0 + 1.0 / D)              # Integrale Coulombiano
    K = np.exp(-D) * (1.0 + D)                                      # Integrale di Scambio
    
    E_elec_LCAO = -0.5 - (J + K) / (1.0 + S)
    E_tot_LCAO = E_elec_LCAO + (1.0 / D)

    plt.plot(distanza_internucleare, E_tot_LCAO, color='lightgreen', marker='x', linestyle='none', markersize=6, label='LCAO')


    plt.xlabel('R [a.u.]')
    plt.ylabel('Energia [a.u.]')
    plt.title('Curva di Dissociazione dell\' H2+')
    plt.xlim(0, 4)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()