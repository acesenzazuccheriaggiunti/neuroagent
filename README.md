# NeuroAgent — Hormone-Driven AI Simulation

A behavioral simulation where an autonomous agent makes decisions based on a biologically-inspired hormonal system. The agent learns over time using a neural network trained with Q-learning.

---

## What it does

The agent lives in a 2D environment, searches for food, rests, and explores. Its behavior is not scripted — it emerges from the interaction of five hormones that influence each other in physiologically realistic ways.

A small MLP (neural network) observes the hormonal state at every tick and selects the best action. It trains continuously via Q-learning with experience replay, getting better at keeping the agent healthy over time.

You can intervene manually from the UI — injecting or suppressing any hormone in real time — and watch how the agent's behavior shifts in response.

---

## Features

- **Biologically-grounded hormone model** — dopamine, cortisol, serotonin, melatonin, adrenaline with mutual interactions and circadian rhythms
- **Neural network decision-making** — MLP trained with Q-learning, no external ML libraries (numpy only)
- **Real-time hormone control** — ±1 / ±10 buttons for each hormone directly in the UI
- **Persistent learning** — weights saved to `brain_weights.json` on exit, reloaded on next run
- **Clinical monitor UI** — color-coded hormone bars, energy readout, epsilon indicator, agent state
- **Optional headless pre-training** — run `train.py` before the UI for a head start

---

## Project structure

```
neuroagent/
├── agent.py          # hormonal state, reward function, act logic
├── brain.py          # MLP + Q-learning + replay buffer (numpy only)
├── environment.py    # food, temperature, day/night cycle
├── logger.py         # tick-by-tick data logger
├── main.py           # headless run + matplotlib plots
├── plots.py          # hormone and environment charts
├── train.py          # pre-training script (no UI)
├── ui.py             # pygame real-time visualization
└── brain_weights.json  # saved weights (auto-generated)
```

---

## Quickstart

```bash
pip install pygame numpy
python ui.py
```

To pre-train the network before opening the UI (recommended):

```bash
python train.py   # ~300 episodes, takes a few seconds
python ui.py
```

To run without UI and generate matplotlib plots:

```bash
python main.py
```

---

## Hormone model

| Hormone | Biological role | Modeled behavior |
|---|---|---|
| **Dopamine** | Reward prediction error | Spikes on unexpected food, drops on failure |
| **Cortisol** | Stress / HPA axis | Rises with failure and hunger; circadian morning peak |
| **Serotonin** | Mood / wellbeing | Sustained by food and rest; suppressed by chronic cortisol |
| **Melatonin** | Circadian rhythm | Rises at night, rapidly suppressed by daylight |
| **Adrenaline** | Fight-or-flight | Short half-life; driven by cortisol via HPA axis |

### Interactions

The hormones do not operate independently. Key interactions modeled:

- **Cortisol ↔ Melatonin** — antagonists: high cortisol suppresses melatonin (stress disrupts sleep)
- **Serotonin → Dopamine** — high serotonin moderates dopamine (hedonic regulation)
- **Cortisol → Serotonin** — chronic stress depletes serotonin (depressive mechanism)
- **Cortisol → Adrenaline** — cortisol stimulates adrenaline release via HPA axis
- **HPA negative feedback** — cortisol above threshold triggers an active brake

### Known simplifications

This is a didactic model, not a clinical simulator. Notable simplifications:

- Energy is a single variable (no glycemia / fat reserves / muscular fatigue distinction)
- Dopamine is modeled as a sustained level, not a phasic spike signal
- Hormone values are dimensionless [0–100], not physiological units
- No receptor saturation or tolerance effects

---

## Neural network

```
Input (6)   →   Hidden (16, ReLU)   →   Output (3 Q-values)
[energy, dopamine, cortisol, serotonin, melatonin, adrenaline]
                                        [rest, explore, search_food]
```

- **Algorithm** — Q-learning with experience replay (buffer size 5000, batch 32)
- **Exploration** — ε-greedy with exponential decay (1.0 → 0.05)
- **Reward** — composite signal: +serotonin +energy −cortisol −adrenaline +dopamine
- **Implementation** — pure numpy, no PyTorch or TensorFlow

The UI shows the current epsilon value with a color indicator:
- 🔴 `> 0.5` — mostly random exploration
- 🟡 `0.15–0.5` — actively learning
- 🟢 `< 0.15` — exploiting learned policy

---

## UI controls

| Element | Description |
|---|---|
| Hormone bars | Live readout of all 5 hormones + energy |
| `−10` `−1` `+1` `+10` buttons | Manual hormone adjustment |
| Agent color | Red = stress/adrenaline · Green = wellbeing · Blue = calm |
| Bottom-left label | Current behavioral state |
| Bottom-right label | Day / night cycle |
| Epsilon line | Network learning progress |
| Close window | Auto-saves weights to `brain_weights.json` |

---

## Coming Soon

- **Clinical scenarios** — preset initial conditions for burnout, insomnia, depression, trauma response
