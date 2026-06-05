import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time  
from model import model


def calcola_laplaciano(psi: torch.Tensor, r_batch: torch.Tensor) -> torch.Tensor:
    gradiente_psi = torch.autograd.grad(
        outputs=psi, 
        inputs=r_batch, 
        grad_outputs=torch.ones_like(psi), 
        create_graph=True,
        retain_graph=True
    )[0] 
    
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
    dispositivo = r_batch.device 
    nucleo_sx = torch.zeros_like(r_batch)
    nucleo_dx = torch.zeros_like(r_batch)
    nucleo_sx[:, 0] = -R_batch.squeeze()
    nucleo_dx[:, 0] = R_batch.squeeze()
    
    dist_nucleo_sx = torch.norm(r_batch - nucleo_sx, dim=-1)
    dist_nucleo_dx = torch.norm(r_batch - nucleo_dx, dim=-1)
    
    V = -1.0 / torch.sqrt(dist_nucleo_sx**2 + epsilon**2) - 1.0 / torch.sqrt(dist_nucleo_dx**2 + epsilon**2)
    return V


def calcola_Loss_PDE(psi: torch.Tensor, r_batch: torch.Tensor, E_corrente: torch.Tensor, R_batch: float) -> torch.Tensor:
    laplaciano = calcola_laplaciano(psi, r_batch)
    V = calcola_potenziale(r_batch, R_batch)
    H_psi = -0.5 * laplaciano + V * psi
    residuo = H_psi - (E_corrente * psi)
    return torch.mean(residuo**2)

def calcola_Loss_BC(psi: torch.Tensor, r_batch: torch.Tensor, R_batch: float) -> torch.Tensor:
    dispositivo = r_batch.device
    nucleo_sx = torch.zeros_like(r_batch)
    nucleo_dx = torch.zeros_like(r_batch)
    nucleo_sx[:, 0] = -R_batch.squeeze()
    nucleo_dx[:, 0] = R_batch.squeeze()

    dist_nucleo_sx = torch.norm(r_batch - nucleo_sx, dim=-1)
    dist_nucleo_dx = torch.norm(r_batch - nucleo_dx, dim=-1)

    soglia_distanza = 9.5 # paper: 17.5
    punti_lontani = (dist_nucleo_sx > soglia_distanza) & (dist_nucleo_dx > soglia_distanza)

    if torch.sum(punti_lontani) == 0:
        return torch.tensor(0.0) 

    psi_lontano = psi[punti_lontani]
    return torch.mean(psi_lontano**2)
    
# TRAINING
if __name__ == '__main__':

    model = model(input_dim=4, n_hidden=2, n_neurons=16)
    ottimizzatore = optim.Adam(model.parameters(), lr=8*10**(-3))

    epoche = 4000
    punti_per_epoca = 15000
    valori_loss = []          
    miglior_loss = float('inf')  
    migliori_pesi = None
    valori_energia = []

    valori_loss_PDE = []
    valori_loss_BC = []

    print("\nInizio Training generale...")
    start_time_globale = time.time()  

    for epoca in range(epoche):
        r_batch = (torch.rand(punti_per_epoca, 3) * 20 ) - 10 
        r_batch.requires_grad_(True) 
        R_batch = (3.0 - 0.2) * torch.rand(punti_per_epoca, 1) + 0.2

        psi, energy = model(r_batch, R_batch)
        loss_PDE = calcola_Loss_PDE(psi, r_batch, energy, R_batch)
        loss_BC = calcola_Loss_BC(psi, r_batch, R_batch)
        loss_totale = loss_PDE + loss_BC  

        loss_totale.backward() 
        ottimizzatore.step()
        ottimizzatore.zero_grad()

        if loss_totale.item() < miglior_loss:
            miglior_loss = loss_totale.item()
            migliori_pesi = copy.deepcopy(model.state_dict())

        valori_loss_PDE.append(loss_PDE.item()) 
        valori_loss_BC.append(loss_BC.item())
        valori_loss.append(loss_totale.item())
        valori_energia.append(torch.mean(energy).item())

    tempo_globale = time.time() - start_time_globale
    minuti = int(tempo_globale // 60)
    secondi = tempo_globale % 60
    print(f"Training totale completato in {minuti} minuti e {secondi:.2f} secondi")

   # Fine-Tuning
    print("\nInizio fase di Fine-Tuning...")
    start_time_ft = time.time()  

    model.load_state_dict(copy.deepcopy(migliori_pesi))
        
    for param in model.basis_unit.parameters():
        param.requires_grad = False
    for param in model.gate.parameters():
        param.requires_grad = False
    
    ottimizzatore_ft = optim.Adam(model.energy_unit.parameters(), lr=1e-3)
    
    epoche_ft = 2000
    for epoca_ft in range(epoche_ft):
        r_batch = (torch.rand(punti_per_epoca, 3) * 20 ) - 10 
        r_batch.requires_grad_(True)
        R_batch_ft = (3.0 - 0.2) * torch.rand(punti_per_epoca, 1) + 0.2

        psi, energy = model(r_batch, R_batch_ft)
        loss_PDE = calcola_Loss_PDE(psi, r_batch, energy, R_batch_ft)
        loss_BC = calcola_Loss_BC(psi, r_batch, R_batch_ft)
        loss_totale = loss_PDE + loss_BC  
        
        loss_totale.backward() 
        ottimizzatore_ft.step()
        ottimizzatore_ft.zero_grad()
        
        if loss_totale.item() < miglior_loss:
            miglior_loss = loss_totale.item()
            migliori_pesi = copy.deepcopy(model.state_dict())

        valori_loss_PDE.append(loss_PDE.item()) 
        valori_loss_BC.append(loss_BC.item())
        valori_loss.append(loss_totale.item())
        valori_energia.append(torch.mean(energy).item())

    tempo_ft = time.time() - start_time_ft
    minuti_ft = int(tempo_ft // 60)
    secondi_ft = tempo_ft % 60
    print(f"Fine-Tuning completato in {minuti_ft} minuti e {secondi_ft:.2f} secondi")

    print("\nTraining completato. Salvataggio dei risultati in corso...")     
    
    # SALVATAGGIO DEI PESI E DEI LOG
    checkpoint = {
        'model_state_dict': migliori_pesi,
        'valori_loss_PDE': valori_loss_PDE,
        'valori_loss_BC': valori_loss_BC,
        'valori_loss': valori_loss,
        'valori_energia': valori_energia,
        'epoche_pre_ft': epoche  # Salvo a che punto è iniziato il fine tuning
    }
    
    torch.save(checkpoint, 'best_model_results.pth')
    print("Salvataggio completato con successo nel file 'best_model_results.pth'!")