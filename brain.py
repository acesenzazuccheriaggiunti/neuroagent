import numpy as np
import random
import json
import os

# ──────────────────────────────────────────────────────────────
# struttura della rete neurale
#
# la rete prende come input lo stato interno dell'agente:
#
#   [energia, dopamina, cortisolo, serotonina, melatonina, adrenalina]
#
# tutti i valori arrivano già normalizzati nell'intervallo [0, 1].
#
# la rete ha:
#   - 6 neuroni di input
#   - 16 neuroni nello strato nascosto
#   - 3 neuroni di output, uno per ogni possibile azione
#
# gli output non sono probabilità: sono Q-values.
# un Q-value rappresenta quanto il Brain considera conveniente
# eseguire una certa azione partendo da un determinato stato.
#
# la rete implementa quindi una versione minimale di
# Q-learning con una rete neurale come funzione approssimatrice.
# ──────────────────────────────────────────────────────────────

ACTIONS = ["rest", "explore", "search_food"]
N_IN = 6
N_HID = 16
N_OUT = len(ACTIONS)


def relu(x):
    # lascia invariati i valori positivi e porta a zero quelli negativi
    return np.maximum(0, x)

class Brain:

    def __init__(self, lr=0.01, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        # learning rate:
        # controlla quanto velocemente i pesi della rete vengono
        # modificati durante il training
        self.lr = lr
        # discount factor:
        # determina quanto il Brain considera importanti le ricompense
        # future rispetto a quella ottenuta immediatamente
        self.gamma = gamma
        # parametri della strategia epsilon-greedy:
        # con epsilon = 1.0 il Brain inizialmente sceglie quasi sempre
        # azioni casuali (esplorazione)
        # durante l'apprendimento epsilon diminuisce, quindi il brain
        # tende progressivamente a sfruttare ciò che ha imparato
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # pesi della rete ──────────────────────────────────
        # W1 collega i 6 input ai 16 neuroni nascosti
        #   W1 = (6, 16)
        # W2 collega i 16 neuroni nascosti ai 3 output
        #   W2 = (16, 3)
        # i pesi vengono inizializzati casualmente
        # la scala sqrt(2 / numero_input) evita valori iniziali troppo grandi o troppo piccoli
        self.W1 = np.random.randn(N_IN,  N_HID) * np.sqrt(2 / N_IN)
        self.b1 = np.zeros(N_HID)
        self.W2 = np.random.randn(N_HID, N_OUT) * np.sqrt(2 / N_HID)
        self.b2 = np.zeros(N_OUT)

        # experience replay ────────────────────────────────
        # il brain non impara solamente dall'ultima esperienza,
        # conserva una serie di esperienze passate e ne estrae
        # casualmente dei piccoli gruppi durante il training
        # ogni esperienza contiene:
        #   (stato, azione, reward, stato_successivo)
        self.memory = []
        # numero di esperienze utilizzate in ogni aggiornamento dei pesi
        self.batch_size = 32

    # forward ────────────────────────────────────────────────

    def forward(self, state):
        # propaga lo stato attraverso la rete
        # primo strato:
        #    h_pre = state · W1 + b1
        # applicazione della ReLU:
        #    h = ReLU(h_pre)
        # secondo strato:
        #    q = h · W2 + b2
        # il risultato q contiene un Q-value per ogni azione:
        #    q[0] → rest
        #    q[1] → explore
        #    q[2] → search_food
        
        self.h_pre = state @ self.W1 + self.b1
        self.h = relu(self.h_pre)
        self.q = self.h @ self.W2 + self.b2
        return self.q

    # scelta azione ──────────────────────────────────────────

    def act(self, state):
        # sceglie un'azione usando la strategia epsilon-greedy
        # con probabilità epsilon:
        #    esplora → sceglie un'azione casuale
        # altrimenti:
        #    sfrutta → sceglie l'azione con Q-value maggiore

        if random.random() < self.epsilon:
            return random.randint(0, N_OUT - 1)   # esplorazione casuale
        q = self.forward(state)
        return int(np.argmax(q))

    def action_name(self, idx):
        # converte l'indice numerico prodotto dalla rete nel nome leggibile dell'azione
        return ACTIONS[idx]

    # memorizza esperienza ───────────────────────────────────

    def remember(self, state, action, reward, next_state):
        # salva una transizione nell'experience replay buffer
        # una transizione rappresenta:
        #    stato
        #      ↓
        #    azione
        #      ↓
        #    ambiente
        #      ↓
        #    reward + nuovo stato
        #i l buffer viene limitato a 5000 esperienze per evitare una crescita indefinita della memoria
        
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 5000:
            self.memory.pop(0)

    # training: Q-learning con replay ───────────────────────

    def train(self):
        # aggiorna i pesi della rete usando Q-learning
        # 1. seleziona un batch casuale di esperienze
        # 2. calcola il Q-value attuale
        # 3. stima il valore desiderato usando reward + futuro
        # 4. calcola l'errore
        # 5. propaga l'errore all'indietro nella rete
        # 6. aggiorna i pesi
        # senza esperienze sufficienti non si esegue nessun aggiornamento
        
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state in batch:
            # Q-value dello stato corrente ──────────────────
            # q_current contiene una stima del valore di tutte
            # le azioni possibili nello stato corrente.
            q_current = self.forward(state).copy()
            # Q-value dello stato successivo ────────────────
            # per Q-learning interessa sapere quale sarebbe
            # la migliore ricompensa futura possibile
            q_next = self.forward(next_state)
            # equazione fondamentale del Q-learning:
            # target = reward + gamma * max(Q(next_state))
            target = reward + self.gamma * np.max(q_next)
            # si copiano i Q-values attuali e si modificano solamente
            # quelli relativi all'azione che è stata realmente eseguita
            q_target = q_current.copy()
            q_target[action] = target

            # backpropagation manuale ───────────────────────
            # si calcola quanto i Q-values attuali differiscono dal target desiderato
            dq = q_current - q_target                          
            # gradiente dei pesi dello strato di output
            dW2 = np.outer(self.h, dq)                          
            db2 = dq
            # propagazione dell'errore dallo strato di output verso lo strato nascosto
            dh = dq @ self.W2.T
            # derivata della ReLU:
            #   x > 0 → 1
            #   x <= 0 → 0
            # i neuroni spenti dalla ReLU non ricevono gradiente.                                
            dh_pre = dh * (self.h_pre > 0).astype(float)    
            # gradiente dei pesi dello strato di input.    
            dW1 = np.outer(state, dh_pre)                      
            db1 = dh_pre

            # aggiornamento dei pesi ────────────────────────
            # si muove ogni peso nella direzione che riduce l'errore
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        # dopo un aggiornamento del brain si riduce gradualmente l'esplorazione casuale
        # epsilon non scende mai sotto epsilon_min
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # salvataggio pesi ────────────────────────────────────

    def save(self, path="brain_weights.json"):
        #salva i parametri appresi:
        #    W1, b1 → primo strato
        #    W2, b2 → secondo strato
        #    epsilon → livello corrente di esplorazione
        
        data = {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "epsilon": self.epsilon
        }
        with open(path, "w") as f:
            json.dump(data, f)

    # caricamento pesi ──────────────────────────────────
    def load(self, path="brain_weights.json"):
        # carica i pesi precedentemente salvati,
        # se il file non esiste, si mantengono i pesi casuali dell'inizializzazione
        
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.W1 = np.array(data["W1"])
        self.b1 = np.array(data["b1"])
        self.W2 = np.array(data["W2"])
        self.b2 = np.array(data["b2"])
        self.epsilon = data.get("epsilon", self.epsilon_min)
        return True