import random

class Environment:

    def __init__(self):

        self.time = 0

        self.is_day = True

        self.food_available = 50

        self.temperature = 20

    def update(self):

        self.time += 1

        # ciclo giorno/notte

        if self.time % 50 == 0:
            self.is_day = not self.is_day

        # variazione temperatura

        self.temperature += random.uniform(-0.5, 0.5)

        # clamp temperatura

        self.temperature = max(-10, min(40, self.temperature))

        # rigenerazione cibo

        self.food_available += random.randint(0, 3)

        self.food_available = min(100, self.food_available)