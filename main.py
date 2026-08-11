from agent import Agent
from environment import Environment
from logger import Logger
from plots import Plots

N_AGENTS = 3
# Ogni agente ha il proprio Brain e quindi il proprio file di pesi.
# In questo modo gli agenti possono imparare indipendentemente.
agents = [
    Agent(name=f"agent_{i}", weights_path=f"brain_weights_{i}.json")
    for i in range(N_AGENTS)
]
env = Environment()
logger = Logger()

# Ciclo principale della simulazione:
# a ogni tick aggiorniamo l'ambiente, facciamo decidere e agire
# gli agenti, aggiorniamo il loro stato interno, registriamo i dati
# e infine calcoliamo le interazioni sociali tra gli agenti.
for tick in range(500):

    env.update()

    for agent in agents:
        agent.decide()
        agent.act(env)
        agent.update_hormones(env)
        logger.log(tick, agent, env)

    env.social_step(agents)

df = logger.get_dataframe()

print(df.head())

# grafici del primo agente (comportamento invariato rispetto alla baseline)
df_agent0 = df[df["agent_id"] == "agent_0"]
Plots.hormones(df_agent0)
Plots.energy(df_agent0)
Plots.environment(df_agent0)