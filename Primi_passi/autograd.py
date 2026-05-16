import torch 

# Pytorch non è solo una libreria per fare operazioni sui tensori (come numpy), 
# ma è un motore di Differenziazione Automatica (da cui auto-grad, automatic gradient).

# Quando si usa numpy, per calcolare una derivata si hanno due strade:

# 1) Analitica (simbolica): come si fa su carta. Applichi le regole di derivazione (es. la derivata di x^2 è 2x).
# 2) Numerica: si usano le differenze finite calcolando il rapporto incrementale con h piccolissimo. Questo metodo, però, introduce errori di arrotondamento enormi nei computer.

# Autograd fa una cosa diversa: ogni volta che fai un'operazione su un tensore che ha "requires_grad=True", pytorch costruisce un "grafo computazionale" invisibile in memoria. 
# Si annota ogni singola addizione, moltiplicazione o esponenziale fatta. Quando gli viene chiesta la derivata, ripercorre questo grafo all'indietro applicando in modo esatto la regola della catena.
# Risultato: restituisce la derivata esatta e in modo velocissimo, senza errori di approssimazione numerica.


# TEST del Laplaciano con funzione Gaussiana

# f(x,y,z) = exp(-alpha*(x^2 +y^2 + z^2))
alpha = 1 # sia alpha=1

def gaussiana (r):       #r^2 = x^2 +y^2 + z^2
    # N.B. la derivata della norma (sqrt(x^2 +y^2 + z^2)) ha singolarità!
    # Ma nella gaussiana appare distanza^2, quindi uso direttamente:
    r_quadrato = torch.sum(r**2, dim=-1) # calcola direttamente x^2 +y^2 + z^2 (le loro derivate non hanno singolarità)
    return torch.exp(-alpha*r_quadrato)

def laplaciano_analitico(r):
    r_quadrato = torch.sum(r**2, dim=-1)
    return 2*alpha* torch.exp(-alpha*r_quadrato)*(2*alpha*r_quadrato-3)

# Autograd per il laplaciano automatico
def laplaciano_autograd(f_val, r):
    # f_val: valore della funzione calcolata (il tensore di output)
    # r: coordinate in ingresso (il tensore di input)
    # 1) Derivata prima: (gradient)
    grad_f = torch.autograd.grad(f_val, r, grad_outputs=torch.ones_like(f_val), create_graph=True)[0]
    # create_graph=True è fondamentale: Mantiene viva la "memoria" delle operazioni matematiche fatte 
    # durante la derivata prima, permettendo di derivare di nuovo quel risultato (per ottenere le derivate seconde)
    # invece grad_outputs=torch.ones_like(f_val) è dovuto al fatto che pyTorch sa calcolare in automatico solo la derivata di un singolo numero (uno scalare).
    # Quindi si scrive F = 1* f_1(r1) + 1*f_2(r2) , e derivando F si ottengono le singole derivate

    laplaciano = 0
    # 2) Derivate seconde: 
    # Il tensore 'r' ha 3 colonne (x, y, z). Faccio un ciclo sulle 3 dimensioni spaziali.
    for i in range (3):
        # derivo la componente i-esima del gradiente rispetto alla coordinata i-esima
        deriv_seconda = torch.autograd.grad(grad_f[:,i], r, grad_outputs=torch.ones_like(grad_f[:, i]), retain_graph=True)[0] # operatore di slicing ':' = significa "prendi tutte.." in questo caso Le righe (poi i=0, solo la colonna delle x)
        # sommo solo la diagonale (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)
        laplaciano += deriv_seconda[:, i]
    return laplaciano

# Corpo codice: (TEST)

# Creo un elettrone in un punto a caso (es. x=0.5, y=0.1, z=-0.2)
# NB: lo creo come matrice 1x3 per simulare un "batch" di 1 elettrone.
r_test = torch.tensor([[0.5, 0.1, -0.2]], requires_grad=True)

valore_f = gaussiana(r_test)

risultato_esatto = laplaciano_analitico(r_test)
risultato_pytorch = laplaciano_autograd(valore_f, r_test)

print(f"Punto di test r: {r_test.tolist()}")
print(f"Valore funzione: {valore_f.item():.5f}")
print(f"Laplaciano (Formula Analitica): {risultato_esatto.item():.8f}")
print(f"Laplaciano (PyTorch Autograd) : {risultato_pytorch.item():.8f}")

errore = torch.abs(risultato_esatto - risultato_pytorch)
print(f"Errore tra i due metodi:        {errore.item():.2e}")