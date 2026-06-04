"""
train.py — pre-addestramento headless (senza pygame)
Esegui prima di ui.py per dare alla rete un punto di partenza decente.

    python train.py

Salva i pesi in brain_weights.json che ui.py carica automaticamente.
"""

from agent import Agent
from environment import Environment

EPISODES = 300
TICKS    = 500

print("Addestramento headless in corso...")

for ep in range(EPISODES):
    agent = Agent()
    env   = Environment()
    agent.brain.load()          # continua da dove si era fermato

    total_reward = 0

    for tick in range(TICKS):
        env.update()
        agent.decide()
        agent.act(env)
        agent.update_hormones(env)
        total_reward += agent.compute_reward()

    agent.brain.save()

    if (ep + 1) % 50 == 0:
        avg = total_reward / TICKS
        eps = agent.brain.epsilon
        print(f"  episodio {ep+1:3d}/{EPISODES}  |  reward medio {avg:.3f}  |  epsilon {eps:.3f}")

print("Addestramento completato. Pesi salvati in brain_weights.json")