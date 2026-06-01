import torch 
from torch import nn
import copy
import matplotlib.pyplot as plt
import numpy as np
from funzione_f import ModelloH2

def calcola_potenziale(r, R):
    R1 = torch.tensor([-R, 0.0, 0.0])
    R2 = torch.tensor([R, 0.0, 0.0])

    # Calcolo le distanze 
    d1 = torch.sqrt(torch.sum((r - R1)**2, dim=-1))
    d2 = torch.sqrt(torch.sum((r - R2)**2, dim=-1))
    
    # Correzione vicino ai nuclei, per evitare singolarità
    epsilon = 0.05
    # così il termine non ha discontinuità di derivata, utile per l’autograd.
    V_coulombiano = -(1 / (torch.sqrt(d1**2 + epsilon**2))) - (1 / (torch.sqrt(d2**2 + epsilon**2)))
    
    return V_coulombiano

def calcola_laplaciano(psi, r):
    # Calcolo del gradiente (restituisce un tensore della stessa forma di r [batch_size, 3])
    grad_psi = torch.autograd.grad(
        outputs=psi, 
        inputs=r, 
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True
    )[0]
    
    laplaciano = torch.zeros_like(psi)
    
    # Calcolo la derivata seconda componente per componente
    for i in range(3):  # for i in range(r.shape[1]): # Cicla su x, y, z
        grad_i = grad_psi[:, i] # Prendi la i-esima colonna del gradiente (es. dPsi/dx)
        
        # Calcola la derivata di (dPsi/dx) rispetto a r (x, y, z)
        # grad_grad_i sarà un tensore [batch_size, 3] contenente (d^2Psi/dx^2, d^2Psi/dxdy, d^2Psi/dxdz)
        grad_grad_i = torch.autograd.grad(
            outputs=grad_i, 
            inputs=r, 
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Mi serve solo la componente diagonale (es. d^2Psi/dx^2), che si trova nella colonna i
        laplaciano += grad_grad_i[:, i]
        
    return laplaciano

if __name__ == '__main__':

    # SIMULAZIONE DI TRAINING

    modello = ModelloH2() # funzione d'onda completa
    # Ottimizzatore Adam
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=8*10**(-3))


    valori_loss = []          
    miglior_loss = float('inf')  
    migliori_pesi = None
    valori_energia = []
    # Ciclo di allenamento 
    epoche = 1500
    for epoca in range(epoche):
        r_punti = torch.rand(2000, 3) * 10.0 - 5.0 # numeri tra 0 e 10 poi -> -5 e 5
        r_batch = r_punti.clone().detach().requires_grad_(True)

        # Il modello restituisce Psi e l'Energia predetta
        psi, E_pred = modello(r_batch)
        
        laplaciano = calcola_laplaciano(psi, r_batch)
        V = calcola_potenziale(r_batch, R=1.0)
        
        # H*Psi
        H_psi = -0.5 * laplaciano + V * psi
        
        # LOSS PDE
        # Bisogna minimizzare il residuo dell'equazione di Schrödinger: (H*Psi - E*Psi)^2
        residuo = H_psi - E_pred * psi
        loss_PDE = torch.mean(residuo**2)
        
        loss_PDE.backward()
        ottimizzatore.step() 
        ottimizzatore.zero_grad()
        
        loss_attuale = loss_PDE.item()
        valori_loss.append(loss_attuale)
        valori_energia.append(modello.E.item())

        if loss_attuale < miglior_loss:
            miglior_loss = loss_attuale
            migliori_pesi = copy.deepcopy(modello.state_dict())
            if epoca % 400 == 0:
                print(f"Epoca {epoca} | Loss PDE: {miglior_loss:.6f} | Energia E: {modello.E.item():.4f}")

    print(f"Training completato. La loss minima raggiunta è stata: {miglior_loss:.6f}")
    modello.load_state_dict(migliori_pesi)

    energia_finale = modello.E.item()
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

    # Disegno di psi_totale
    asse_x = torch.linspace(-4, 4, 1000)
    r_punti = torch.zeros(1000, 3) # tensore 3D. Ora y e z rimangono zero
    r_punti[:, 0] = asse_x

    with torch.no_grad():
        # Calcolo della Psi_LCAO
        psi_lcao = modello.phi_1.valuta(r_punti) + modello.phi_2.valuta(r_punti)
        # Calcolo del modello totale
        psi_totale, _ = modello(r_punti)

        f = modello.funzione_f(r_punti)   
        val_nn_pos = modello.rete(r_punti)
        val_nn_neg = modello.rete(-r_punti)
        N_simmetrica = 0.5 * (val_nn_pos + val_nn_neg)
        
        correzione_neurale = f * N_simmetrica
        N = N_simmetrica

        # passaggio a numpy per usare matplot
        x_np = asse_x.numpy()
        lcao_np = psi_lcao.numpy()
        totale_np = psi_totale.numpy()
        neurale_np = correzione_neurale.numpy()
        
        # Prevenire la divisione per zero con numpy
        f_numpy = f.numpy()
        N_np = np.divide(neurale_np, f_numpy, out=np.zeros_like(neurale_np), where=f_numpy!=0)

        # Normalizzo x il confronto visivo
        lcao_norm = lcao_np / np.max(np.abs(lcao_np))
        totale_norm = totale_np / np.max(np.abs(totale_np))

        plt.figure(figsize=(10, 6))
        plt.plot(x_np, lcao_norm, label='Psi LCAO (Normalizzata)', linestyle='--', color='gray', linewidth=2)
        plt.plot(x_np, totale_norm, label='Psi PINN (Normalizzata)', color='blue', linewidth=2)
        
        # Scalare la correzione N per farla entrare nel grafico senza sballare l'asse Y
        N_scalata = N_np / np.max(np.abs(N_np)) * 0.5 if np.max(np.abs(N_np)) > 0 else N_np
        plt.plot(x_np, N_scalata, label='Correzione N (Scalata)', color='red', alpha=0.5)

        # nuclei
        plt.axvline(x=-1.0, color='red', linestyle=':', alpha=0.5, label='Nuclei')
        plt.axvline(x=1.0, color='red', linestyle=':', alpha=0.5)
        plt.title("Confronto 1D: LCAO Classica vs PINN")
        plt.xlabel("x (a.u.)")
        plt.ylabel("Ampiezza Relativa")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()