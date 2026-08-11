# train.py — pre-addestramento headless del brain
# esegue la simulazione senza interfaccia grafica per permettere
# agli agenti di accumulare esperienze e aggiornare i propri pesi
# prima dell'esecuzione con la UI
# esecuzione:
#    python train.py

from agent import Agent
from environment import Environment

EPISODES = 300
TICKS = 500
N_AGENTS = 3   # numero di agenti indipendenti da addestrare in parallelo

print("Addestramento headless in corso...")

# ogni episodio rappresenta una nuova simulazione
for ep in range(EPISODES):
    # creiamo gli agenti con brain indipendenti
    agents = [
        Agent(name=f"agent_{i}", weights_path=f"brain_weights_{i}.json")
        for i in range(N_AGENTS)
    ]
    # ogni episodio parte da un nuovo ambiente
    env = Environment()
    for a in agents:
        a.brain.load(a.weights_path)   # continua da dove si era fermato

    total_reward = [0] * N_AGENTS

    # ciclo principale dell'addestramento
    for tick in range(TICKS):
        env.update()
        for idx, a in enumerate(agents):
            # il brain osserva lo stato e sceglie un'azione
            a.decide()
            # l'agente esegue l'azione e modifica il proprio stato
            a.act(env)
            # viene aggiornata la componente fisiologica dell'agente
            a.update_hormones(env)
            # viene accumulato il reward ottenuto in questo tick per valutare l'andamento dell'episodio
            total_reward[idx] += a.compute_reward()
        # dopo le azioni individuali si aggiornano le interazioni sociali tra gli agenti
        env.social_step(agents)

    for a in agents:
        a.brain.save(a.weights_path)

    # viene mostrato un riepilogo ogni 50 episodi
    if (ep + 1) % 50 == 0:
        avg = sum(total_reward) / (N_AGENTS * TICKS)
        eps = agents[0].brain.epsilon
        print(f"  episodio {ep+1:3d}/{EPISODES}  |  reward medio {avg:.3f}  |  epsilon {eps:.3f}")

print(f"Addestramento completato. Pesi salvati in brain_weights_0..{N_AGENTS-1}.json")