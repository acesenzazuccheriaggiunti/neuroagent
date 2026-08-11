import random
import math

class Environment:

    def __init__(self):
        # tempo simulato
        self.time = 0
        # stato giorno/notte per modulare il ritmo circadiano
        self.is_day = True
        # quantità condivisa di cibo disponibile nell'ambiente
        self.food_available = 50
        # temperatura ambientale corrente
        self.temperature = 20

    def update(self):
        # aggiorna lo stato dell'ambiente di un tick
        # l'ambiente evolve indipendentemente dalle decisioni dei singoli agenti
        self.time += 1

        # ciclo giorno/notte ───────────────────────────────
        # ogni 50 tick si inverte lo stato.
        if self.time % 50 == 0:
            self.is_day = not self.is_day

        # temperatura ──────────────────────────────────────
        # piccola variazione casuale per evitare che la temperatura rimanga perfettamente costante
        self.temperature += random.uniform(-0.5, 0.5)

        # la temperatura deve rimanere all'interno dell'intervallo previsto dal modello
        self.temperature = max(-10, min(40, self.temperature))

        # rigenerazione del cibo ───────────────────────────
        # ad ogni tick l'ambiente può generare una piccola quantità
        # di nuovo cibo, la risorsa condivisa non può superare 100
        self.food_available += random.randint(0, 3)

        self.food_available = min(100, self.food_available)

    # interazioni sociali ───────────────────────────────────
    # l'ambiente gestisce le relazioni spaziali tra gli agenti
    # l'interazione rimane in Agent.social_interact()
    # gli agenti consumano la stessa risorsa condivisa in Agent.act()

    def social_step(self, agents):
        # calcola le interazioni sociali tra tutte le coppie di agenti
        # per ogni coppia:
        # 1. si calcola la distanza;
        # 2. si fa interagire A con B;
        # 3. si fa interagire B con A.
        # l'interazione è bidirezionale perché lo stato di ciascun
        # agente può influenzare quello dell'altro
        
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                # distanza euclidea tra i due agenti:
                # sqrt((x1-x2)^2 + (y1-y2)^2)
                d = math.hypot(a.x - b.x, a.y - b.y)
                
                a.social_interact(b, d)
                b.social_interact(a, d)