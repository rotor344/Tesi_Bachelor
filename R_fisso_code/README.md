# R_fisso_code

Questa cartella contiene l'implementazione e gli script per effettuare un addestramento semplificato della PINN. 

## Dettagli del Training Ridotto
L'obiettivo di questo codice è testare il funzionamento e l'accuratezza della rete prima di passare a un campionamento più esteso. Nello specifico:
* Le coordinate spaziali dell'elettrone $r$ vengono campionate in un intervallo ridotto compreso tra -10 e 10.
* La distanza internucleare $R$ non è trattata come una variabile di input casuale, ma è mantenuta come un parametro fisso per tutta la durata dell'addestramento[cite: 1, 6].
* L'energia totale del sistema è implementata come un parametro scalare della rete neurale, che viene ottimizzato durante il processo di training.

## Modelli Implementati a Step
Al fine di poter confrontare i risultati e l'accuratezza delle funzioni d'onda ottenute al variare dell'architettura, l'implementazione è stata suddivisa in step progressivi. All'avvio del codice di training è possibile scegliere quale modello testare:

* **Modello 1 (Completamente Blind):** La rete non ha alcun bias fisico iniziale e la funzione d'onda è interamente appresa dalla rete neurale ($\Psi_{tot} = N$).
* **Modello 2 (LCAO + Correzione):** Viene introdotta la base fisica; la rete agisce come semplice termine additivo alla funzione d'onda classica ($\Psi_{tot} = \Psi_{LCAO} + N$).
* **Modello 3 (LCAO + Gate):** La correzione della rete neurale viene modulata da un'apposita rete Gate ($\Psi_{tot} = \Psi_{LCAO} + f \circ N$).
* **Modello 4 (Simmetria Esplicita):** Un'evoluzione del modello precedente in cui la simmetria di parità dello stato fondamentale viene forzata esplicitamente all'interno dell'architettura.