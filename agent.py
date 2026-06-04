import random
import numpy as np
from brain import Brain

class Agent:

    def __init__(self):

        self.energy     = 100
        self.dopamine   = 50
        self.cortisol   = 20
        self.serotonin  = 50
        self.melatonin  = 10
        self.adrenaline = 5

        self.state      = "idle"
        self.brain      = Brain()
        self.brain.load()           # carica pesi se esistono

        self._last_state  = None
        self._last_action = None

    # ── stato normalizzato per la rete ────────────────────────

    def get_state_vector(self):
        return np.array([
            self.energy     / 100,
            self.dopamine   / 100,
            self.cortisol   / 100,
            self.serotonin  / 100,
            self.melatonin  / 100,
            self.adrenaline / 100,
        ], dtype=np.float32)

    # ── reward: quanto sta "bene" l'agente ora ────────────────

    def compute_reward(self):
        reward  =  self.serotonin  * 0.3   # benessere
        reward +=  self.energy     * 0.3   # sopravvivenza
        reward -=  self.cortisol   * 0.2   # penalità stress
        reward -=  self.adrenaline * 0.1   # penalità allerta
        reward +=  self.dopamine   * 0.1   # motivazione
        return reward / 100                # normalizza ~[-1, 1]

    # ── decide: la rete sceglie l'azione ─────────────────────

    def decide(self):
        s = self.get_state_vector()

        # se abbiamo uno step precedente, alleniamo
        if self._last_state is not None:
            reward = self.compute_reward()
            self.brain.remember(self._last_state, self._last_action, reward, s)
            self.brain.train()

        action_idx  = self.brain.act(s)
        self.state  = self.brain.action_name(action_idx)

        self._last_state  = s
        self._last_action = action_idx

    # ── act ───────────────────────────────────────────────────

    def act(self, env):

        if self.state == "rest":
            self.energy     += 5
            self.melatonin  -= 4
            self.cortisol   -= 3
            self.adrenaline  = max(0, self.adrenaline - 3)
            self.serotonin  += 1

        elif self.state == "explore":
            surprise = random.random()
            if surprise > 0.6:
                self.dopamine += random.uniform(1, 4)
            else:
                self.dopamine += random.uniform(-1, 0.5)

        elif self.state == "search_food":

            if env.food_available > 0:
                if random.random() < 0.6:
                    food_amount = random.randint(5, 15)
                    env.food_available -= food_amount
                    self.energy     += food_amount
                    self.adrenaline  = max(0, self.adrenaline - 2)
                    self.dopamine   += random.uniform(4, 10)
                    self.serotonin  += 2
                    self.cortisol    = max(0, self.cortisol - 1)
                else:
                    self.cortisol   += 4
                    self.adrenaline += 3
                    self.energy     -= 5
                    self.dopamine   -= random.uniform(1, 3)
            else:
                self.cortisol   += 6
                self.adrenaline += 4
                self.energy     -= 8
                self.dopamine   -= 2
                self.serotonin  -= 1

    # ── update_hormones ───────────────────────────────────────

    def update_hormones(self, env):

        # ritmo circadiano melatonina
        if env.is_day:
            self.melatonin -= 0.5
        else:
            self.melatonin += 2

        # cortisolo circadiano
        hour_in_cycle = env.time % 50
        if env.is_day:
            if hour_in_cycle < 8:
                self.cortisol += 1.5
            else:
                self.cortisol -= 0.3
        else:
            self.cortisol -= 0.5

        # interazioni reciproche
        if self.cortisol > 60:
            self.melatonin -= (self.cortisol - 60) * 0.05
        if self.melatonin > 50:
            self.cortisol  -= (self.melatonin - 50) * 0.02
        if self.serotonin > 60:
            self.dopamine  -= (self.serotonin - 60) * 0.02
        if self.dopamine > 70:
            self.serotonin -= (self.dopamine  - 70) * 0.015
        if self.cortisol > 50:
            self.serotonin -= (self.cortisol  - 50) * 0.03

        # adrenalina ↔ cortisolo
        if self.cortisol > 40:
            self.adrenaline += (self.cortisol - 40) * 0.02
        self.adrenaline *= 0.92

        # feedback negativo asse HPA
        if self.cortisol > 75:
            self.cortisol -= (self.cortisol - 75) * 0.05

        # decay naturale
        self.dopamine  *= 0.99
        self.cortisol  *= 0.98
        self.serotonin *= 0.995

        # fatica
        self.energy -= 0.2

        # clamp
        self.dopamine   = max(0, min(100, self.dopamine))
        self.cortisol   = max(0, min(100, self.cortisol))
        self.serotonin  = max(0, min(100, self.serotonin))
        self.melatonin  = max(0, min(100, self.melatonin))
        self.adrenaline = max(0, min(100, self.adrenaline))
        self.energy     = max(0, min(100, self.energy))