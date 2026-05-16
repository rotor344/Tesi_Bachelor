import torch 
from torch import nn
from basis_unit import NeuralCorrection
from AU_torch_version import SingleWavefunction
import copy
import matplotlib.pyplot as plt
import numpy as np
from funzione_f import ModelloH2
from L_PDE_R_fisso import calcola_laplaciano, calcola_potenziale

class EnergyUnit(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )
        
        nn.init.constant_(self.net[-1].bias, -1.0)

    def forward(self, R_batch):
        E = self.net(R_batch)
        return E.squeeze(-1)
    
# Evoluto
class ModelloH2(nn.Module):
    def __init__(self):
        super().__init__()
        # Rete per la forma dell'onda
        self.rete_onda = NeuralCorrection()
        # Rete per l'energia
        self.rete_energia = EnergyUnit()

        self.R_val = 1.0 # fisso i nuclei
        self.phi_1 = SingleWavefunction(-self.R_val, 0.0, 0.0)
        self.phi_2 = SingleWavefunction(self.R_val, 0.0, 0.0)


    def funzione_f(self, r):
        r_quadrato = torch.sum(r**2, dim=-1) 
        return torch.exp(-0.2 * r_quadrato) 
    
    # N.B. r è la matrice (N, 3) degli elettroni, R_batch è la matrice (N, 1) delle distanze
    def forward(self, r):
        
        # Parte SPAZIALE
        psi_lcao = self.phi_1.valuta(r) + self.phi_2.valuta(r)
        f = self.funzione_f(r)
        
        # Calcolo la correzione N(r)
        valore_nn_dritta = self.rete_onda(r)
        valore_nn_rovesciata = self.rete_onda(-r)
        valore_nn_symm = 0.5 * (valore_nn_dritta + valore_nn_rovesciata)
        # Costruiamo la psi totale
        psi_totale = psi_lcao + f*valore_nn_symm
        
        # Part ENERGETICA 
        # Poiché r è (N, 3), creo un vettore R_batch (N, 1) tutto riempito col valore di R fisso
        R_batch = torch.ones(r.shape[0], 1, device=r.device) * self.R_val
        # Chiedo alla rete dell'energia quanto vale E per queste distanze R
        E_pred = self.rete_energia(R_batch)
        
        return psi_totale, E_pred
    

if __name__ == '__main__':

    # SIMULAZIONE DI TRAINING

    modello = ModelloH2() # funzione d'onda completa

    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=8*10**(-3))


    valori_loss = []           
    miglior_loss = float('inf') 
    migliori_pesi = None
    valori_energia = []

    # PHASE 1 : training completo
    # Ciclo di allenamento 
    epoche = 1500
    for epoca in range(epoche):
        r_core = torch.rand(4000, 3) * 10.0 - 5.0   # 80% dei punti attorno alla molecola
        r_tails = torch.rand(1000, 3) * 24.0 - 12.0 # 20% dei punti nel vuoto cosmico
        r_punti = torch.cat((r_core, r_tails), dim=0) # Li unisco in un unico batch da 5000

        r_batch = r_punti.clone().detach().requires_grad_(True)

        psi, E_pred = modello(r_batch)
        
        laplaciano = calcola_laplaciano(psi, r_batch)
        V = calcola_potenziale(r_batch, R=1.0)
        
        # LOSS PDE 
        H_psi = -0.5 * laplaciano + V * psi
        residuo = H_psi - E_pred * psi
        loss_PDE = torch.mean(residuo**2)
        
        # LOSS BC (Condizioni al contorno-boundary)
        r_cut = 10 # oltre 10 a.u. dai nuclei, la psi deve essere zero
        r_norm = torch.norm(r_batch, dim=-1) # valuto distanza dall'origine dei 5000 elettroni 
        mask_bc = r_norm > r_cut # True: punti fuori dal raggio di cutoff , False: interni

        if mask_bc.any(): # se ce n'è almeno uno True
            # media dei valori di psi valutata nei punti oltre r_cutoff (al quadrato)
            loss_BC = torch.mean(psi[mask_bc]**2) 
        else: # tutti false, quindi fuori da r_cut la loss_BC deve essere già 0
            loss_BC = torch.tensor(0.0, device=r_batch.device)
        
        # LOSS TOTALE 
        loss_totale = loss_PDE + loss_BC

        loss_totale.backward()
        ottimizzatore.step() 
        ottimizzatore.zero_grad()
        
        loss_attuale = loss_totale.item()
        valori_loss.append(loss_attuale)

        E_media = torch.mean(E_pred).item()
        valori_energia.append(E_media)

        if loss_attuale < miglior_loss:
            miglior_loss = loss_attuale
            migliori_pesi = copy.deepcopy(modello.state_dict())
            if epoca % 50 == 0:
                E_media = torch.mean(E_pred).item()
                print(f"Epoca {epoca} | Loss: {miglior_loss:.6f} | Energia E: {E_media:.4f}")
    
    modello.load_state_dict(migliori_pesi)
    # PHASE 2: fine-tuning
    print("Inizio Fase 2: Fine-Tuning dell'Energia...")
    
    # Congelamento della rete spaziale
    for parametro in modello.rete_onda.parameters():
        parametro.requires_grad = False
            
    # L'Energy Unit NON la congelo (i suoi parametri restano requires_grad=True)

    # Creo un nuovo Adam che guarda solo ai parametri che non sono congelati
    ottimizzatore_finetune = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modello.parameters()), 
        lr=1e-4)
    
    epocs = 1000
    for epoc in range(epocs):
        r_core = torch.rand(4000, 3) * 10.0 - 5.0   # 80% dei punti attorno alla molecola
        r_tails = torch.rand(1000, 3) * 24.0 - 12.0 # 20% dei punti nel vuoto cosmico
        r_punti = torch.cat((r_core, r_tails), dim=0) # Li unisco in un unico batch da 5000
         
        r_batch = r_punti.clone().detach().requires_grad_(True)
        # Il modello restituisce Psi e l'Energia predetta
        psi, E_pred = modello(r_batch)            
        laplaciano = calcola_laplaciano(psi, r_batch)
        V = calcola_potenziale(r_batch, R=1.0)
            
        # # LOSS PDE
        H_psi = -0.5 * laplaciano + V * psi
        residuo = H_psi - E_pred * psi
        loss_PDE = torch.mean(residuo**2)
        
        # LOSS BC (Condizioni al contorno-boundary)
        r_cut = 10 # oltre 10 a.u. dai nuclei, la psi deve essere zero
        r_norm = torch.norm(r_batch, dim=-1) # Distanza dall'origine
        mask_bc = r_norm > r_cut # Trova i punti fuori dal raggio di cutoff

        if mask_bc.any():
            # Penalizza pesantemente se la psi è diversa da zero in quei punti
            loss_BC = torch.mean(psi[mask_bc]**2) 
        else:
            loss_BC = torch.tensor(0.0, device=r_batch.device)
        
        # LOSS TOTALE 
        loss_totale = loss_PDE + loss_BC

        loss_totale.backward()
        ottimizzatore_finetune.step()
        ottimizzatore_finetune.zero_grad()

        loss_attuale = loss_totale.item()
        valori_loss.append(loss_attuale)
        
        E_media = torch.mean(E_pred).item()
        valori_energia.append(E_media)

        if loss_attuale < miglior_loss:
            miglior_loss = loss_attuale
            migliori_pesi = copy.deepcopy(modello.state_dict())

            print(f"Epoca {epoc+1500} | Loss: {miglior_loss:.6f} | Energia E: {E_media:.4f}")


    print(f"Training completato. La loss minima raggiunta è stata: {miglior_loss:.6f}")
   
    modello.load_state_dict(migliori_pesi)
    with torch.no_grad():
        R_test = torch.tensor([[1.0]])
        E_finale = modello.rete_energia(R_test).item()
        
    print(f"L'Energia finale stimata dalla PINN per R=1.0 è: {E_finale:.6f} a.u.\n")
    
    # DISEGNO dei grafici

    # Grafico della loss function
    plt.figure(figsize=(8, 5))
    plt.plot(valori_loss, color='blue', linewidth=2)
    # Linea per segnare il passaggio tra le due fasi
    plt.axvline(x=1500, color='gray', linestyle=':', label='Inizio Fine-Tuning')
    plt.xlabel('Epoca')
    plt.ylabel('Loss PDE')
    plt.grid(True, alpha=0.3)
    plt.title('Andamento della Loss')
    plt.legend()
    plt.show()

    # grafico dell'Energia
    plt.figure(figsize=(8, 5))
    plt.plot(valori_energia, color='green', linewidth=2, label='Energia stimata dal PINN')
    plt.axvline(x=1500, color='gray', linestyle=':', label='Inizio Fine-Tuning')
    
    plt.xlabel('Epoca')
    plt.ylabel('Energia (a.u.)')
    plt.grid(True, alpha=0.3)
    plt.title("Convergenza dell'Energia")
    plt.legend()
    plt.show()

    # Disegno di psi_totale (come prima)
    asse_x = torch.linspace(-4, 4, 1000)
    r_punti = torch.zeros(1000, 3) 
    r_punti[:, 0] = asse_x

    with torch.no_grad():
        psi_lcao = modello.phi_1.valuta(r_punti) + modello.phi_2.valuta(r_punti)
        psi_totale, _ = modello(r_punti)

        f = modello.funzione_f(r_punti)

        val_nn_pos = modello.rete_onda(r_punti)
        val_nn_neg = modello.rete_onda(-r_punti)
        N_simmetrica = 0.5 * (val_nn_pos + val_nn_neg)
        
        correzione_neurale = f * N_simmetrica
        N = N_simmetrica

        x_np = asse_x.numpy()
        lcao_np = psi_lcao.numpy()
        totale_np = psi_totale.numpy()
        neurale_np = correzione_neurale.numpy()
        
        f_numpy = f.numpy()
        N_np = np.divide(neurale_np, f_numpy, out=np.zeros_like(neurale_np), where=f_numpy!=0)

        lcao_norm = lcao_np / np.max(np.abs(lcao_np))
        totale_norm = totale_np / np.max(np.abs(totale_np))

        plt.figure(figsize=(10, 6))
        plt.plot(x_np, lcao_norm, label='Psi LCAO (Normalizzata)', linestyle='--', color='gray', linewidth=2)
        plt.plot(x_np, totale_norm, label='Psi PINN (Normalizzata)', color='blue', linewidth=2)
        
        N_scalata = N_np / np.max(np.abs(N_np)) * 0.5 if np.max(np.abs(N_np)) > 0 else N_np
        plt.plot(x_np, N_scalata, label='Correzione N (Scalata)', color='red', alpha=0.5)

        plt.axvline(x=-1.0, color='red', linestyle=':', alpha=0.5, label='Nuclei')
        plt.axvline(x=1.0, color='red', linestyle=':', alpha=0.5)
        plt.title("Confronto 1D: LCAO Classica vs PINN (Dopo Fine-Tuning)")
        plt.xlabel("x (a.u.)")
        plt.ylabel("Ampiezza Relativa")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    