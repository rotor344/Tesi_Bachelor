import torch 
from torch import nn
from AU_torch_version import SingleWavefunction
import copy # per fotografare i pesi della loss minore
import matplotlib.pyplot as plt

class NeuralCorrection(nn.Module):
    def __init__(self):
        # Inizializzazione della classe "padre" nn.Module
        super().__init__()

        # BASIS UNIT (BU) 
        # Input layer: prende le 3 coordinate dell'elettrone (x, y, z) e le porta a 16 neuroni
        self.bu_layer1 = nn.Linear(3, 16)
        # Secondo layer nascosto: da 16 neuroni a 16 neuroni
        self.bu_layer2 = nn.Linear(16, 16)

        # GATE 
        # Un layer da 16 neuroni (output della BU) a 10 neuroni
        self.gate_layer = nn.Linear(16, 10)

        # OUTPUT 
        # L'ultimo passaggio deve per forza schiacciare i 10 neuroni del gate in un 
        # singolo numero (il valore della correzione N(r, R) per quel punto)
        self.output_layer = nn.Linear(10, 1)

        # Definisco la funzione di attivazione (citata nel paper )
        self.attivazione = nn.Sigmoid()

    # FORWARD PASS
    def forward(self, r): 
        # 1) Passaggio nella Basis Unit
        x = self.bu_layer1(r)
        x = self.attivazione(x)  # Applicare la sigmoide
        x = self.bu_layer2(x)
        x = self.attivazione(x)
        # 2) Passaggio nel Gate
        x = self.gate_layer(x)
        x = self.attivazione(x)
        # 3) Output finale (nessuna sigmoide alla fine, la correzione può essere positiva o negativa)
        correzione= self.output_layer(x)

        # Per far combaciare perfettamente le dimensioni, occorre "schiacciare" eventuali parentesi di troppo
        return correzione.squeeze(-1) # il layer di output restituisce una matrice con una singola colonna [[valore1], [valore2]]. 
                                      # Lo squeeze toglie le parentesi quadre interne trasformandolo in un vettore pulito [valore1, valore2], perfetto per essere sommato alla LCAO
    


# Questa classe conterrà al suo interno sia i due nuclei, sia la rete neurale.
class ModelloH2(nn.Module):
    def __init__(self):
        super().__init__()
        # Fisso i nuclei
        R = 1
        self.phi_1 = SingleWavefunction(-R, 0.0, 0.0)
        self.phi_2 = SingleWavefunction(R, 0.0, 0.0)
        # Inizializzo il "cervello" 
        self.rete = NeuralCorrection()

    def forward(self, r):
        # Calcolo LCAO
        valore_lcao = self.phi_1.valuta(r) + self.phi_2.valuta(r)
        # Calcolo N(r)
        valore_nn = self.rete(r)
        # Unisco (per ora ometto f(R) per mantenere tutto semplicissimo)
        psi_totale = valore_lcao + valore_nn
        return psi_totale

if __name__ == '__main__':
    # SIMULAZIONE DI TRAINING

    modello = ModelloH2() # funzione d'onda completa

    # Scelgo l'Ottimizzatore citato nel paper (Adam)
    ottimizzatore = torch.optim.Adam(modello.parameters(), lr=8*10**(-3))
    # model.parameters() dice ad Adam di guardare dentro ModelloH2, trovare tutti i pesi della NeuralCorrection e prepararsi a modificarli

    valori_loss = [] # Lista vuota per il grafico finale           
    miglior_loss = float('inf')  # Partendo da infinito la prima epoca vince per forza
    migliori_pesi = None
    # Ciclo di training 
    epoche = 1000
    for epoca in range(epoche):
        # Campionamento (generare elettroni a caso nello spazio)
        r_batch = torch.randn(1000, 3, requires_grad=True)
        # Forward Pass (calcolare la Psi per tutti gli elettroni)
        psi = modello(r_batch)
        # Calcolo dell'errore (Loss funcion)
        loss = torch.mean(psi**2) # Esempio fittizio: per ora la loss è un numero a caso da minimizzare
        # Calcolo delle Derivate (Backward Pass)
        loss.backward() # pytorch calcola le derivate della Loss rispetto a tutti i pesi della rete
        # Aggiornamento dei Pesi
        ottimizzatore.step() # Adam modifica i pesi della rete neurale
        ottimizzatore.zero_grad()  # Reset delle memorie per il giro successivo

        # registrazione dati 
        loss_attuale = loss.item()
        valori_loss.append(loss_attuale)
        # Check
        if loss_attuale < miglior_loss:
            miglior_loss = loss_attuale
            # Fare una fotografia profonda (deepcopy) dello stato attuale della rete (state_dict)
            migliori_pesi = copy.deepcopy(modello.state_dict())
            print(f"Epoca {epoca} | Nuovo record! Loss= {miglior_loss:.6f}")
        

    print(f"Training completato. La loss minima raggiunta è stata: {miglior_loss:.6f}")
    # Ora che il training è finito, bisogna dire al modello di utilizzare
    # i pesi dell'epoca migliore, scartando quelli dell'ultima epoca
    modello.load_state_dict(migliori_pesi)

    # disegno della loss function
    plt.plot(valori_loss, color='blue', linewidth=2)
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.title('Andamento della Loss')
    plt.yscale('log')
    plt.show()