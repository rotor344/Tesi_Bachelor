# Progetto di tesi Bachelor
Pievaioli Davide 911544 
 

## Obiettivo Principale
TBD

## Architettura di Base

Gli script sono organizzati in ordine cronologico. Consiglio di leggerli in questo ordine per una comprensione coerente del progetto.

## Ordine di lettura consigliato

1. `atomic_unit.py` / `AU_torch_version.py`: fondamenti delle funzioni d'onda atomiche
2. `autograd.py`: comprensione del calcolo automatico dei gradienti
3. `basis_unit.py`: architettura della rete neurale di correzione
4. `funzione_f.py`: funzione d'onda totale con fattore di smorzamento
5. `L_PDE_R_fisso.py`: operatori differenziali (Laplaciano e potenziale)
6. `L_TOT_R_fisso.py`: modello completo con rete per l'energia


### `atomic_unit.py` 
`atomic_unit.py` : Funzioni d'onda atomiche (numpy)

Definisce le funzioni d'onda atomiche di base usando numpy. Contiene la classe `SingleWaveFunction` che rappresenta una funzione d'onda atomica centrata su un nucleo, i cui metodi principali sono:

- `valuta()`: calcola il valore della funzione d'onda $\phi = e^{-r}$ in una posizione data
- `disegna()`: genera visualizzazioni 1D e 3D della funzione d'onda

### `AU_torch_version.py`
`AU_torch_version.py` : Funzioni d'onda atomiche (pytorch)

Reimplementa le funzioni d'onda atomiche con PyTorch. La classe `SingleWavefunction` memorizza la posizione del nucleo come tensore torch e fornisce:

- `valuta()`: calcolo vettorizzato della funzione d'onda
- `disegna_1d()` e `disegna_3d()`

Rispetto alla versione numpy, questa scrittura consente il calcolo automatico dei gradienti tramite autograd, ed è completamente integrata con le operazioni delle reti neurali.

### `autograd.py`
`autograd.py` : Tutorial sulla differenziazione automatica

Lo script illustra come pytorch costruisce un grafo computazionale durante le operazioni e applica automaticamente la regola della catena. Essenziale per comprendere come vengono calcolati i gradienti necessari all'equazione di Schrödinger durante il training

### `basis_unit.py`
`basis_unit.py` : Sviluppo della rete neurale

Contiene l'architettura della rete neurale che apprende la correzione alla funzione d'onda LCAO.

La classe `NeuralCorrection` implementa:

- Basis Unit: 2 hidden layer con 16 neuroni e attivazione sigmoid
- Gate: 1 layer che riduce i 16 neuroni a 10
- Output: 1 neurone senza attivazione 

`funzione_f()`: Calcola il fattore di smorzamento 

- Parametro  `E`: energia del sistema come parametro addestrabile (iniziale: -1.0)

La simmetria della molecola è incorporata nella rete con il calcolo della media tra $N(r)$ e $N(-r)$.


- Calcola il potenziale Coulombiano affetto dai due nuclei:
$$V(r) = -\frac{1}{\sqrt{d_1^2 + \epsilon^2}} - \frac{1}{\sqrt{d_2^2 + \epsilon^2}}$$
- Aggiunto $\epsilon = 0.05$ per evitare singolarità nei gradienti

### `L_PDE_R_fisso.py` 
`L_PDE_R_fisso.py` : Calcolo degli operatori differenziali

Operatori sono essenziali per formulare l'equazione di Schrödinger nel modello computazionale
Classe `ModelloH2` (versione finale):
- `self.rete_onda`: rete neurale per la forma della funzione d'onda 

### `L_TOT_R_fisso.py`
`L_TOT_R_fisso.py` : Modello Completo con energia e fine-tuning

Contiene il modello finale che combina una rete per la forma della funzione d'onda con una rete dedicata al calcolo dell'energia.

La classe `EnergyUnit` è una rete neurale semplice (1 -> 16 -> 1) che prende come input la distanza internucleare R e restituisce l'energia totale E. Il bias finale è inizializzato a -1.0 come vincolo fisico.

La classe `ModelloH2` versione finale integra:

- `rete_onda`: rete neurale per la forma della funzione d'onda
- `rete_energia`: rete neurale per calcolare l'energia in funzione di R
- Metodo `forward()`: calcola simultaneamente $\Psi(r)$ e $E(R)$



