import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time  
import matplotlib.pyplot as plt
from model_v0 import modello_v0
from modello_paper import modello_paper
from model_v0 import modello_v2

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
    soglia_distanza = 9.5 # 17.5 #
    punti_lontani = (dist_nucleo_sx > soglia_distanza) & (dist_nucleo_dx > soglia_distanza)

    if torch.sum(punti_lontani) == 0:
        return torch.tensor(0.0)  # Nessun punto lontano, nessuna penalizzazione

    psi_lontano = psi[punti_lontani]
    
    # Penalizzo se psi è troppo grande nei punti lontani
    return torch.mean(psi_lontano**2)
    
# TRAINING
if __name__ == '__main__':

    print("Training del modello con architecture come paper")

    #model = modello_v0(input_dim=3, n_hidden=2, n_neurons=16)
    #model = modello_v1(input_dim=2, n_hidden=2, n_neurons=16)
    model = modello_v2(input_dim=4, n_hidden=2, n_neurons=16)

    ottimizzatore = optim.Adam(model.parameters(), lr=8*10**(-3))

    epoche = 4000
    punti_per_epoca = 15000
    valori_loss = []          
    miglior_loss = float('inf')  
    migliori_pesi = None
    valori_energia = []

    valori_loss_PDE = []
    valori_loss_BC = []

    print("\n--- Inizio Training Globale ---")
    start_time_globale = time.time()  # <- Inizio contatore Training Globale

    for epoca in range(epoche):
        
        r_batch = (torch.rand(punti_per_epoca, 3) * 20 ) - 10 # distribuz. tra -10 e +10 a.u.
        #r_batch = (torch.rand(punti_per_epoca, 3) * 36 ) - 18 
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

        if loss_totale.item() < miglior_loss:
            miglior_loss = loss_totale.item()
            migliori_pesi = copy.deepcopy(model.state_dict())

        # salvataggio dati (loss ed energia)
        valori_loss_PDE.append(loss_PDE.item()) 
        valori_loss_BC.append(loss_BC.item())
        valori_loss.append(loss_totale.item())
        valori_energia.append(torch.mean(energy).item())

    # <-- Fine contatore Training Globale
    tempo_globale = time.time() - start_time_globale
    print(f"Training Globale completato in {tempo_globale:.2f} secondi ({tempo_globale/60:.2f} minuti)")

   # Fine-Tuning
    print("\nInizio fase di Fine-Tuning ")
    start_time_ft = time.time()  # <-- Inizio contatore Fine-Tuning

    # Carico i pesi migliori trovati
    model.load_state_dict(copy.deepcopy(migliori_pesi))
        
    # CONGELO Basis Unit e Gate (Nessun aggiornamento pesi)
    for param in model.basis_unit.parameters():
        param.requires_grad = False
    for param in model.gate.parameters():
        param.requires_grad = False
    
    # ottimizzatore esclusivo per la Energy Unit
    # Learning Rate molto più basso (es. 1e-3 o 5e-4) per la "rifinitura"
    ottimizzatore_ft = optim.Adam(model.energy_unit.parameters(), lr=1e-3)
    
    
    epoche_ft = 2000
    for epoca_ft in range(epoche_ft):
        # campionamento 
        r_batch = (torch.rand(punti_per_epoca, 3) * 20 ) - 10 
        #r_batch = (torch.rand(punti_per_epoca, 3) * 36 ) - 18
        r_batch.requires_grad_(True)
        R_batch_ft = (3.0 - 0.2) * torch.rand(punti_per_epoca, 1) + 0.2

        # Forward pass 
        psi, energy = model(r_batch, R_batch_ft)
        
        loss_PDE = calcola_Loss_PDE(psi, r_batch, energy, R_batch_ft)
        loss_BC = calcola_Loss_BC(psi, r_batch, R_batch_ft)
        loss_totale = loss_PDE + loss_BC  
        
        # Backpropagation
        loss_totale.backward() 
        ottimizzatore_ft.step()
        ottimizzatore_ft.zero_grad()
        
        if loss_totale.item() < miglior_loss:
            miglior_loss = loss_totale.item()
            migliori_pesi = copy.deepcopy(model.state_dict())

        # salvataggio dati (loss ed energia)
        valori_loss_PDE.append(loss_PDE.item()) 
        valori_loss_BC.append(loss_BC.item())
        valori_loss.append(loss_totale.item())
        valori_energia.append(torch.mean(energy).item())

    # <-- Fine contatore Fine-Tuning
    tempo_ft = time.time() - start_time_ft
    print(f"Fine-Tuning completato in {tempo_ft:.2f} secondi ({tempo_ft/60:.2f} minuti)")

   # Training COMPLETATO
    print("\nTraining completato ")     
   # Carico i pesi finali migliori
    model.load_state_dict(copy.deepcopy(migliori_pesi))

    # GRAFICI
    '''
    Grafico delle Loss 
    '''
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

    '''
    Grafico dell'Energia E(R)
    '''
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

    '''
    Grafici delle funzioni d'onda — NORMALIZZAZIONE L^2 3D stile paper

    Il paper costruisce psi su una griglia 3D, calcola N = 1 / sqrt( integrale 3D di |psi|^2 ) via
    regola di Simpson iterata (scipy.integrate.simps), e POI fa il taglio 1D ad altissima risoluzione
    lungo y = z = 0.
    '''

    try:
        from scipy.integrate import simpson as _simpson
    except ImportError:
        from scipy.integrate import simps as _simpson

    def calcola_costante_norm_3D(model, R_val, n_grid=60, box=10.0):
        """
        Calcola il Fattore di normalizzazione N integrando su una griglia 3D sparsa.
        """
        coord = torch.linspace(-box, box, n_grid)
        xg, yg, zg = torch.meshgrid(coord, coord, coord, indexing='ij')
        
        pts = torch.stack([xg.flatten(), yg.flatten(), zg.flatten()], dim=-1)
        Rt  = torch.full((pts.shape[0], 1), R_val, dtype=torch.float32)

        with torch.no_grad():
            psi_, _    = model(pts, Rt)
            psi_lcao_  = model.calcola_LCAO(pts, Rt)

        psi      = psi_.reshape(n_grid, n_grid, n_grid).numpy()
        psi_lcao = psi_lcao_.reshape(n_grid, n_grid, n_grid).numpy()
        coord_np = coord.numpy()

        def integrale_simpson_3d(f_3d, assi):
            I = _simpson(_simpson(_simpson(f_3d, x=assi, axis=0), x=assi, axis=0), x=assi, axis=0)
            return float(I)

        vol_pinn = integrale_simpson_3d(psi**2, coord_np)
        vol_lcao = integrale_simpson_3d(psi_lcao**2, coord_np)

        if vol_pinn < 1e-10: vol_pinn = 1.0
        if vol_lcao < 1e-10: vol_lcao = 1.0

        return 1.0 / np.sqrt(vol_pinn), 1.0 / np.sqrt(vol_lcao)

    def estrai_slicing_denso_1D(model, R_val, norm_psi, norm_lcao, n_punti=1000, box=10.0):
        """
        Genera la riga 1D perfetta e ad altissima risoluzione per il grafico.
        Valuta il modello solo sull'asse X (y=0, z=0), moltiplicando poi per N.
        """
        asse_x = torch.linspace(-box, box, n_punti)
        pts_1D = torch.zeros(n_punti, 3)
        pts_1D[:, 0] = asse_x
        Rt_1D = torch.full((n_punti, 1), R_val, dtype=torch.float32)

        with torch.no_grad():
            psi_, _   = model(pts_1D, Rt_1D)
            psi_lcao_ = model.calcola_LCAO(pts_1D, Rt_1D)

        psi_n      = psi_ * norm_psi
        psi_lcao_n = psi_lcao_ * norm_lcao

        return asse_x.numpy(), psi_n.numpy(), psi_lcao_n.numpy()

    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for R_val, ax, color in [(1.0, ax1, 'blue'), (2.0, ax2, 'purple')]:
        # Troviamo i fattori di scala fisici corretti
        N_pinn, N_lcao = calcola_costante_norm_3D(model, R_val=R_val, n_grid=60, box=10.0)
        
        # Creiamo il grafico liscio
        x_axis, psi_cut, lcao_cut = estrai_slicing_denso_1D(
            model, R_val=R_val, norm_psi=N_pinn, norm_lcao=N_lcao, n_punti=1000, box=10.0
        )

        ax.plot(x_axis, psi_cut,  color=color, linewidth=2.5,
                label='Psi PINN (norm 3D)')
        ax.plot(x_axis, lcao_cut, color='gray', linestyle='--', linewidth=2,
                label='Psi LCAO (norm 3D)')
        ax.scatter([-R_val, R_val], [0, 0], color='red', s=50, zorder=5,
                   label='Nuclei')
        ax.axvline(x=-R_val, color='red', linestyle=':', alpha=0.5)
        ax.axvline(x=+R_val, color='red', linestyle=':', alpha=0.5)
        
        ax.set_title(f"Funzione d'onda lungo l'asse internucleare (R = {R_val} a.u.)")
        ax.set_ylabel('Ampiezza normalizzata')
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax2.set_xlabel('x [a.u.]')
    plt.tight_layout()
    plt.show()