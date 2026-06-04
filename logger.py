class Logger:

    def __init__(self):
        self.logs = []

    def log(self, tick, agent, env):

        self.logs.append({

            "tick": tick,

            # agente
            "state": agent.state,
            "energy": agent.energy,
            "dopamine": agent.dopamine,
            "cortisol": agent.cortisol,
            "serotonin": agent.serotonin,
            "melatonin": agent.melatonin,

            # ambiente
            "food_available": env.food_available,
            "temperature": env.temperature,
            "is_day": env.is_day
        })

    def get_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.logs)