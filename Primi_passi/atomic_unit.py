import numpy as np 
import matplotlib.pyplot as plt

"""  
Architecture:
 2 hidden layers of 16 neurons each for the BU
 1 layer of 10 neurons for the gate
 2 layers of 32 neurons each for the EU
 with a sigmoid activation for all the hidden neurons.
 We train the NN using the Adam optimizer with a learning rate of 8 ×10−3. 
 The network is optimized for 5 ×10^3 epochs but we save the model with the lowest L. 
 For the fine-tuning phase we load the best model and train only the EU using the Adam optimizer
 with a lr = 10^-4.
 Anche il campionamento è parte del training
 Nel paper i punti sono campionati nel dominio (x,y,z) in [-18, 18] e R in [0.2, 3]
 con r_cutoff = 17.5 
 e vengono ricampionati casualmente ad ogni epoca.
 N.B. I nuclei sono posti a +/- R -> la distanza interatomica è 2R

 """
# inizio

# ATOMIC UNIT 

# Comincio dalle Phi 
class SingleWaveFunction:
    def __init__ (self, R_x, R_y, R_z):
        self.R_x = R_x
        self.R_y = R_y
        self.R_z = R_z
    
    # funzione per calcolare la funzione d'onda 
    def valuta(self, r_x, r_y, r_z):
        # distanza tra elettrone e questo nucleo
        dx = np.abs(r_x - self.R_x)
        dy = np.abs(r_y - self.R_y)
        dz = np.abs(r_z - self.R_z)

        distanza = np.sqrt(dx**2 + dy**2 + dz**2)

        phi = np.exp(-distanza)
        return phi 
    
    def disegna(self, pedice):
        asse_x = np.linspace(-18, 18, np.pow(10,6))
        asse_y = 0
        asse_z = 0
        valori_phi = np.zeros(len(asse_x))
        
        #for r_x, r_y, r_z in asse_x : 
        for i in range(len(asse_x)):
            valori_phi[i] = self.valuta(asse_x[i], 0, 0)
        plt.plot(asse_x, valori_phi, label=rf'$\Phi_{{{pedice}}}$')
        
        indx_max = np.argmax(valori_phi)     # indice del massimo
        x_max = asse_x[indx_max]             # posizione x del massimo

        plt.axvline(x=x_max, linestyle='--', color='red', alpha=0.3, label=f'max {pedice}')
        #plt.show()  
    
    def disegna_3d(self, ax, titolo): 
        # meno punti rispetto al 1D per non appesantire il 3D
        asse_x = np.linspace(-18, 18, np.pow(10,3))
        asse_y = np.linspace(-18, 18, np.pow(10,3))
        
        # griglia 2D
        X, Y = np.meshgrid(asse_x, asse_y)
        
        # valuto la funzione su tutta la griglia contemporaneamente (Z=0)
        Z = self.valuta(X, Y, 0)
        
        # disegno superficie
        superficie = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title(titolo)
        ax.set_xlabel('x (a.u.)')
        ax.set_ylabel('y (a.u.)')
        ax.set_zlabel('Valore funzione')
        return superficie

# MAIN: (corpo codice)

# Esempio:
# Creo la funzione d'onda per un nucleo che si trova nell'origine (0, 0, 0)
phi_1 = SingleWaveFunction(0.0, 0.0, 0.0)

# Calcolo il suo valore se l'elettrone si trova nel punto (1, 0, 0)
valore = phi_1.valuta(1.0, 0.0, 0.0)
print("Valore della funzione d'onda:", valore)

# Caso reale (con R=1):
phi_1 = SingleWaveFunction(-1, 0, 0)
phi_2 = SingleWaveFunction(1, 0, 0)

phi_1.disegna(1)
phi_2.disegna(2)
plt.xlabel("x (a.u.)")
plt.ylabel("Valori della funzione")
plt.legend()
plt.title('Single wavefunctions per R=1')
plt.show()


# Esempio 3D 
phi_1 = SingleWaveFunction(-1, 0, 0)
phi_2 = SingleWaveFunction(1, 0, 0)

# Configuro la figura 3D di matplot
fig = plt.figure(figsize=(12, 5))

# sottografico per la phi 1
ax1 = fig.add_subplot(121, projection='3d')
phi_1.disegna_3d(ax1, r"$\Phi_1$ centrata in $x=-1$")

# sottografico per la phi 2
ax2 = fig.add_subplot(122, projection='3d')
phi_2.disegna_3d(ax2, r"$\Phi_2$ centrata in $x=1$")

plt.tight_layout()
plt.show()

# per disegnarle insieme (alla fine è LCAO)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# creo un'unica griglia condivisa
asse_x = np.linspace(-4, 4, 100)
asse_y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(asse_x, asse_y)

# valuto entrambe le funzioni sulla stessa griglia (sempre nel piano z=0)
Z1 = phi_1.valuta(X, Y, 0)
Z2 = phi_2.valuta(X, Y, 0)

# creo l'orbitale molecolare sommando le matrici
Z_tot = Z1 + Z2

# disegno la superficie risultante
superficie = ax.plot_surface(X, Y, Z_tot, cmap='viridis', edgecolor='none', alpha=0.9)

# estetica
ax.set_title(r"Orbitale Molecolare legante di $H_2^+$ ($\Phi_1 + \Phi_2$)")
ax.set_xlabel('x [a.u.]')
ax.set_ylabel('y [a.u.]')
ax.set_zlabel('Ampiezza funzione d\'onda')

# barra dei colori per capire i valori
fig.colorbar(superficie, shrink=0.5, aspect=10, label='Ampiezza')

plt.show()


# LCAO 
# ground state solo asse x (solo stato legante, col +)

asse_x_lcao = np.linspace(-18, 18, 1000)

# Valuto le due funzioni d'onda separate lungo l'asse x [ R = 1]
valori_phi1 = phi_1.valuta(asse_x_lcao, 0, 0)
valori_phi2 = phi_2.valuta(asse_x_lcao, 0, 0)

# stati legati LCAO
psi_lcao_plus = valori_phi1 + valori_phi2
# psi_lcao_minus = valori_phi1 - valori_phi2 # stato antilegante, energia maggiore

# Disegno
plt.figure(figsize=(8, 5))
plt.plot(asse_x_lcao, psi_lcao_plus, label=r'$\Psi_{LCAO+} = \Phi_1 + \Phi_2$ (GS)', color='purple', linewidth=2)

# funzioni originarie per confronto (tratteggiate)
plt.plot(asse_x_lcao, valori_phi1, linestyle='--', label=r'$\Phi_1$', color='blue', alpha=0.5)
plt.plot(asse_x_lcao, valori_phi2, linestyle='--', label=r'$\Phi_2$', color='red', alpha=0.5)

# posizione dei nuclei
plt.axvline(x=-1, color='gray', linestyle=':', label='Nucleo 1')
plt.axvline(x=1, color='gray', linestyle=':', label='Nucleo 2')

plt.title("Controllo LCAO Ground State lungo l'asse x")
plt.xlabel("x [a.u.]")
plt.ylabel("Ampiezza funzione d'onda")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# verifica della simmetria per x=+/- 0.5 
psi_LCAO_dx = phi_1.valuta(0.5, 0, 0) + phi_2.valuta(0.5, 0, 0)
psi_LCAO_sx = phi_1.valuta(-0.5, 0, 0) + phi_2.valuta(-0.5, 0, 0)

print('Verifica della simmetria di Psi_LCAO: \n' ,
      '.\n' , '.\n' ,
      'differenza della funz valutata in x=0.5 e x=-0.5:', f'{psi_LCAO_dx - psi_LCAO_sx }'
      )




    