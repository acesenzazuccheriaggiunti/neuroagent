import numpy as np
import random
import json
import os

# ──────────────────────────────────────────────────────────────
#  MLP minimale — pesi in numpy puro, niente dipendenze esterne
#  Input:  6 valori normalizzati [0,1]
#          (energy, dopamine, cortisol, serotonin, melatonin, adrenaline)
#  Hidden: 16 neuroni, ReLU
#  Output: 3 Q-values  (rest, explore, search_food)
# ──────────────────────────────────────────────────────────────

ACTIONS = ["rest", "explore", "search_food"]
N_IN    = 6
N_HID   = 16
N_OUT   = len(ACTIONS)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Brain:

    def __init__(self, lr=0.01, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # pesi — Xavier init
        self.W1 = np.random.randn(N_IN,  N_HID) * np.sqrt(2 / N_IN)
        self.b1 = np.zeros(N_HID)
        self.W2 = np.random.randn(N_HID, N_OUT) * np.sqrt(2 / N_HID)
        self.b2 = np.zeros(N_OUT)

        # replay buffer
        self.memory     = []
        self.batch_size = 32

    # ── forward ────────────────────────────────────────────────

    def forward(self, state):
        self.h_pre = state @ self.W1 + self.b1
        self.h     = relu(self.h_pre)
        self.q     = self.h @ self.W2 + self.b2
        return self.q

    # ── scelta azione ──────────────────────────────────────────

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, N_OUT - 1)   # esplorazione casuale
        q = self.forward(state)
        return int(np.argmax(q))

    def action_name(self, idx):
        return ACTIONS[idx]

    # ── memorizza esperienza ───────────────────────────────────

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 5000:
            self.memory.pop(0)

    # ── training: Q-learning con replay ───────────────────────

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state in batch:
            q_current  = self.forward(state).copy()
            q_next     = self.forward(next_state)
            target      = reward + self.gamma * np.max(q_next)
            q_target    = q_current.copy()
            q_target[action] = target

            # backprop manuale
            dq  = q_current - q_target                          # (N_OUT,)
            dW2 = np.outer(self.h, dq)                          # (N_HID, N_OUT)
            db2 = dq
            dh  = dq @ self.W2.T                                # (N_HID,)
            dh_pre = dh * (self.h_pre > 0).astype(float)        # ReLU grad
            dW1 = np.outer(state, dh_pre)                       # (N_IN, N_HID)
            db1 = dh_pre

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ── salva / carica pesi ────────────────────────────────────

    def save(self, path="brain_weights.json"):
        data = {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "epsilon": self.epsilon
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path="brain_weights.json"):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.W1      = np.array(data["W1"])
        self.b1      = np.array(data["b1"])
        self.W2      = np.array(data["W2"])
        self.b2      = np.array(data["b2"])
        self.epsilon = data.get("epsilon", self.epsilon_min)
        return True