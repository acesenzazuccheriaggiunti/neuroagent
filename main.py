from agent import Agent
from environment import Environment
from logger import Logger
from plots import Plots

agent = Agent()
env = Environment()
logger = Logger()

for tick in range(500):

    env.update()

    agent.decide()
    agent.act(env)
    agent.update_hormones(env)

    logger.log(tick, agent, env)

df = logger.get_dataframe()

print(df.head())

Plots.hormones(df)
Plots.energy(df)
Plots.environment(df)