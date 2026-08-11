class Logger:

    def __init__(self):
        # lista delle osservazioni raccolte durante la simulazione
        # ogni elemento sarà un dizionario contenente lo stato
        # dell'agente e dell'ambiente in un determinato tick
        self.logs = []

    def log(self, tick, agent, env):
        self.logs.append({

            "tick": tick,

            # agente
            "agent_id": getattr(agent, "name", "agent"),
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
        # converte lo storico in un DataFrame Pandas
        import pandas as pd
        return pd.DataFrame(self.logs)