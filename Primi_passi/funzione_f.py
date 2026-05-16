import torch 
from torch import nn
from basis_unit import NeuralCorrection
from AU_torch_version import SingleWavefunction
import copy
import matplotlib.pyplot as plt

class ModelloH2(nn.Module):
    def __init__(self):
        super().__init__()
        R = 1.0 # fisso i nuclei
        self.phi_1 = SingleWavefunction(-R, 0.0, 0.0)
        self.phi_2 = SingleWavefunction(R, 0.0, 0.0)
        self.rete = NeuralCorrection()
        
        # L'Energia è un parametro scalare addestrabile (lo inizializzo a -1.0)
        self.E = nn.Parameter(torch.tensor([-1.0])) 
        """ 
        Senza questo guess, la rete neurale per l'energia partirebbe di default da un valore molto vicino allo zero. 
        In MQ un'energia E>=0 per la molecola di idrogeno non descrive uno stato di legame stabile, ma uno stato del continuo, 
        ovvero un elettrone ionizzato e "libero" che se ne va a spasso. 
        Se la rete iniziasse la sua ricerca da energie positive, si ritroverebbe in un "altopiano" matematico e 
        farebbe una fatica immane a "vedere" e scendere nella buca di potenziale profonda e stretta che rappresenta il legame covalente. 
        """

    def funzione_f(self, r):
        r_quadrato = torch.sum(r**2, dim=-1) 
        return torch.exp(-0.2 * r_quadrato) 
    
    # exp^(-abs(r)) avrebbe cuspide in r= 0 -> derivata prima e seconda 'pericolose' invece la gaussiana è perfettamente liscia (differenziabile infinite volte)
    def forward(self, r):
        valore_lcao = self.phi_1.valuta(r) + self.phi_2.valuta(r)
        # incorporo la simmetria fisica
        valore_nn_dritta = self.rete(r)
        valore_nn_rovesciata = self.rete(-r)
        valore_nn_simmetrico = 0.5 * (valore_nn_dritta + valore_nn_rovesciata)
        
        f = self.funzione_f(r)
        psi_totale = valore_lcao + (f * valore_nn_simmetrico)
        
        # Ritorno sia Psi che E 
        return psi_totale, self.E
    
if __name__ == '__main__':

    # SIMULAZIONE DI TRAINING

    modello = ModelloH2() # funzione d'onda completa

    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=8*10**(-3))

    valori_loss = []           
    miglior_loss = float('inf')  
    migliori_pesi = None
    # Ciclo di allenamento 
    epoche = 1000
    for epoca in range(epoche):
        # Campionamento (generare elettroni a caso nello spazio) 
        r_batch = torch.randn(1000, 3, requires_grad=True)
        # Forward Pass (calcolare la Psi per tutti gli elettroni)
        psi, _ = modello(r_batch)
        # Calcolo della Loss
        loss = torch.mean(psi**2) # Per ora la loss è un numero a caso da minimizzare
        # Calcolo delle Derivate (Backward Pass)
        loss.backward() 
        # Aggiornamento dei pesi
        ottimizzatore.step() 
        ottimizzatore.zero_grad()  
        # registrazione dati 
        loss_attuale = loss.item()
        valori_loss.append(loss_attuale)
        # Check
        if loss_attuale < miglior_loss:
            miglior_loss = loss_attuale
            # fotografia profonda (deepcopy) dello stato attuale della rete (state_dict)
            migliori_pesi = copy.deepcopy(modello.state_dict())
            print(f"Epoca {epoca} | Nuovo record! Loss= {miglior_loss:.6f}")

        if epoca % 200 == 0:
            print(f"Epoca {epoca} | Loss fittizia= {loss.item():.4f}")
        

    print(f"Training completato. La loss minima raggiunta è stata= {miglior_loss:.6f}")

    modello.load_state_dict(migliori_pesi)

    # disegno della loss function
    '''plt.plot(valori_loss, color='blue', linewidth=2)
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.title('Andamento della Loss')
    plt.show()'''

    # Disegno di psi_totale
    asse_x = torch.linspace(-4, 4, 1000)
    r_punti = torch.zeros(1000, 3) # tensore 3D. Ora y e z rimangono zero
    r_punti[:, 0] = asse_x

    with torch.no_grad():
        # Calcolo della Psi_LCAO (fisica)
        psi_lcao = modello.phi_1.valuta(r_punti) + modello.phi_2.valuta(r_punti)
        # Calcolo del modello (LCAO + NN)
        psi_totale, _ = modello(r_punti)
        # Isolo contributo della rete neurale
        f = modello.funzione_f(r_punti)
        correzione_neurale = f * modello.rete(r_punti)
        N = correzione_neurale / f
    # passaggio a numpy per usare matplot
    x_np = asse_x.numpy()
    lcao_np = psi_lcao.numpy()
    totale_np = psi_totale.numpy()
    neurale_np = correzione_neurale.numpy()
    N_np = N.numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x_np, lcao_np, label='Psi LCAO (Pura)', linestyle='--', color='gray', linewidth=2)
    plt.plot(x_np, neurale_np, label='Correzione Neurale (f * N)', color='orange', alpha=0.6)
    plt.plot(x_np, N_np, label='Correzione Neurale senza f (N)', color='red', alpha=0.6)
    plt.plot(x_np, totale_np, label='Psi Totale (Loss fittizia)', color='blue', linewidth=2)
    # nuclei
    plt.axvline(x=-1.0, color='red', linestyle=':', alpha=0.5, label='Nuclei')
    plt.axvline(x=1.0, color='red', linestyle=':', alpha=0.5)
    plt.title("Confronto 1D: LCAO Classica vs PINN (Loss fittizia)")
    plt.xlabel("x (a.u.)")
    plt.ylabel("Ampiezza Funzione d'onda")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

