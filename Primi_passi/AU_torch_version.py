import torch 
import numpy as np
import matplotlib.pyplot as plt

# invece di passare r_x, r_y, r_z come variabili separate, in pytorch 
# conviene passare un unico vettore posizione r (un tensore di 3 elementi). 
# Questo perché per calcolare l'energia cinetica, dovrò calcolare
# la derivata della funzione d'onda rispetto al vettore posizione dell'elettrone"

class SingleWavefunction:
    def __init__(self, R_x, R_y, R_z):
        # Salvo la posizione del nucleo come un tensore 1D di 3 elementi.
        self.R = torch.tensor([R_x, R_y, R_z], dtype=torch.float32) # dtype=torch.float32 è lo standard per le reti neurali.
    
    def valuta(self, r):
        # 'r' è il tensore posizione dell'elettrone, ad esempio: tensor([1.0, 0.0, 0.0])
        distanza = torch.norm(r - self.R, dim=-1)
        # Se, come nel caso del training, passassi un batch (lotto) di N elettroni insieme,
        # il tensore 'r' diventerà una matrice 2D (N righe, 3 colonne). 
        # La dimensione 0 (dim=0) sono le 1000 righe (i diversi elettroni).
        # La dimensione 1 (dim=1, che è anche l'ultima, quindi dim=-1) sono le 3 colonne (x,y,z).
        # scrivendo dim=-1, sto andando riga per riga, prendo solo i 3 numeri di quell'elettrone (x,y,z), e calcolo la distanza".

        # Invece di np.exp, uso torch.exp
        phi = torch.exp(-distanza)
        return phi
    
    def disegna_1d(self, pedice):
        # 10000 punti lungo l'asse x usando pytorch
        asse_x = torch.linspace(-18, 18, np.pow(10, 4))
        # bisgona creare una matrice con 10k righe e 3 colonne [x, y, z]
        r_punti = torch.zeros(np.pow(10, 4), 3)
        # inserisco i valori di asse_x nella colonna 0 (ovvero le x)
        for j in range (len(asse_x)):
            r_punti[j, 0] = asse_x[j]
        #r_punti[:, 0] = asse_x  (Scritto meglio in una riga) 

        # Valuto la funzione passandole tutti i punti contemporaneamente
        valori_phi = self.valuta(r_punti)
        # PONTE TRA PYTORCH E MATPLOTLIB
        # estraggo i numeri puri dai tensori di pytorch per darli a matplotlib
        x_numpy = asse_x.numpy()
        y_numpy = valori_phi.numpy()

        plt.plot(x_numpy, y_numpy, label=rf'$\Phi_{{{pedice}}}$')
        # riga verticale dove si trova il massimo (la posizione del nucleo)
        nucleo_x = self.R[0].item() # .item() estrae un singolo numero dal tensore
        plt.axvline(x=nucleo_x, linestyle='--', color='grey', alpha=0.3)

    def disegna_3d(self,ax, pedice):
        asse_x = torch.linspace(-18,18,10000)
        asse_y = torch.linspace(-18,18,10000)
        # griglia 2D
        X, Y = torch.meshgrid(asse_x, asse_y, indexing='xy') # NB: indexing='xy' serve per dire a pytorch di comportarsi esattamente come numpy

        Z_coord = torch.zeros_like(X) # crea una matrice piena di zeri con la stessa identica forma di X
        # impacchetto le tre matrici in un unico tensore 10k x10k x3
        r_punti = torch.stack([X, Y, Z_coord], dim=-1) # dim=-1 : unisce queste matrici mettendole l'una dietro l'altra nell'ultima dimensione
        # Quindi la forma del tesnore diventa (10000, 10000, 3)
        # valuto la funzione d'onda su tutta la griglia
        valori_phi = self.valuta(r_punti)
        # PONTE TRA PYTORCH E MATPLOTLIB
        X_np = X.numpy()
        Y_np = Y.numpy()
        Z_np = valori_phi.numpy() 

        superficie = ax.plot_surface(X_np, Y_np, Z_np, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title(pedice)
        ax.set_xlabel('x [a.u.]')
        ax.set_ylabel('y [a.u.]')
        ax.set_zlabel("Ampiezza")
        return superficie

# MAIN: 
if __name__ == '__main__':
    # Caso in cui R = 1
    R = 1
    phi_1 = SingleWavefunction(-R, 0, 0)
    phi_2 = SingleWavefunction(R, 0, 0)

    # sia un elettrone di prova in x=0.5, y=0, z=0
    r_electron = torch.tensor([0.5, 0,0], requires_grad=True) # requires_grad=True traccia tutte le operazioni fatte su questo
                                                            # punto (perché più tardi chiederò di calcolarne le derivate)
    valore_phi_1 = phi_1.valuta(r_electron)
    valore_phi_2 = phi_2.valuta(r_electron)

    # LCAO
    psi_LCAO = valore_phi_1 + valore_phi_2

    print(f"Valore di Psi_LCAO nel punto {r_electron.tolist()}: {psi_LCAO.item():.5f}")

    # Verifica di simmetria veloce:
    r_simmetrico = torch.tensor([-0.5, 0.0, 0.0])
    psi_simm = phi_1.valuta(r_simmetrico) + phi_2.valuta(r_simmetrico)
    print(f"Valore simmetrico in x=-0.5: {psi_simm.item():.5f}")
    print(f"Conferma della simmetria (differenza): {(psi_LCAO.item()-psi_simm.item()):.5f} ")

    # Disegni
    #2d
    plt.figure(figsize=(8, 4))
    phi_1.disegna_1d(1)
    phi_2.disegna_1d(2)

    plt.title("Orbitali Atomici isolati [versione pytorch]")
    plt.xlabel("x [a.u.]")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    #3d (singola wavefunction)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    phi_1.disegna_3d(ax, r"$\Phi_1$ centrata in $x=-1$ [versione pytorch]")

    plt.show()

    # LCAO 3d
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    asse_x = torch.linspace(-18, 18, 10000)
    asse_y = torch.linspace(-18, 18, 10000)
    X, Y = torch.meshgrid(asse_x, asse_y, indexing='xy')
    Z_coord = torch.zeros_like(X)
    r_punti = torch.stack([X, Y, Z_coord], dim=-1)
    Z1 = phi_1.valuta(r_punti)
    Z2 = phi_2.valuta(r_punti)
    Z_tot = Z1 + Z2 # ground state molecolare
    # pytorch -> numpy
    X_np = X.numpy()
    Y_np = Y.numpy()
    Z_tot_np = Z_tot.numpy()

    superficie = ax.plot_surface(X_np, Y_np, Z_tot_np, cmap='magma', alpha=0.9) 
    ax.set_title(r"Orbitale Molecolare LCAO GS ($\Phi_1 + \Phi_2$) [versione pytorch]")
    ax.set_xlabel('x [a.u.]')
    ax.set_ylabel('y [a.u.]')
    ax.set_zlabel("Ampiezza funzione d'onda")
    fig.colorbar(superficie, shrink=0.5, aspect=10, label='Ampiezza')

    plt.show()