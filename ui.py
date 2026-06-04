import pygame
import random
import math

from agent import Agent
from environment import Environment

pygame.init()

WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NeuroAgent — Monitor Clinico")
clock = pygame.time.Clock()

font_small = pygame.font.SysFont("Courier New", 13)
font_med   = pygame.font.SysFont("Courier New", 15, bold=True)
font_large = pygame.font.SysFont("Courier New", 18, bold=True)

BG         = (12, 14, 22)
BG_CARD    = (24, 28, 42)
BORDER     = (40, 48, 70)
TEXT_PRI   = (220, 228, 255)
TEXT_SEC   = (120, 130, 160)
GREEN      = (52, 211, 153)
RED        = (248, 113, 113)
YELLOW     = (251, 191, 36)
FOOD_COLOR = (74, 222, 128)

SIM_W   = 600
PANEL_X = SIM_W + 10
PANEL_W = WIDTH - SIM_W - 20

HORMONE_CFG = [
    {"key": "dopamine",   "label": "Dopamina",   "desc": "Prediction error / Reward", "color": (167, 139, 250)},
    {"key": "cortisol",   "label": "Cortisolo",  "desc": "Stress / Ritmo circadiano", "color": (248, 113, 113)},
    {"key": "serotonin",  "label": "Serotonina", "desc": "Benessere / Umore",         "color": (52, 211, 153)},
    {"key": "melatonin",  "label": "Melatonina", "desc": "Ritmo circadiano / Sonno",  "color": (96, 165, 250)},
    {"key": "adrenaline", "label": "Adrenalina", "desc": "Fight-or-flight (HPA)",     "color": (251, 191, 36)},
]

agent  = Agent()
env    = Environment()
agent.x = SIM_W // 2
agent.y = HEIGHT // 2

foods   = []
trail   = []
tick    = 0
buttons = []


def distance(ax, ay, bx, by):
    return math.sqrt((ax - bx)**2 + (ay - by)**2)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def spawn_food():
    if len(foods) < 25 and random.random() < 0.1:
        foods.append([random.randint(20, SIM_W - 20), random.randint(20, HEIGHT - 20)])

def agent_color():
    happy  = agent.serotonin  / 100
    stress = agent.cortisol   / 100
    alert  = agent.adrenaline / 100
    hunger = 1 - agent.energy / 100
    r = clamp(int(255 * stress + 100 * alert + 50 * hunger), 0, 255)
    g = clamp(int(220 * happy), 0, 255)
    b = clamp(int(200 * (1 - stress) * (1 - alert)), 0, 255)
    return (r, g, b)

def draw_bar(surface, x, y, w, h, value, color, bg=(40, 48, 70)):
    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=3)
    fill_w = int(w * clamp(value, 0, 100) / 100)
    if fill_w > 0:
        pygame.draw.rect(surface, color, (x, y, fill_w, h), border_radius=3)

def draw_button(surface, rect, label, hovered=False):
    col = (50, 60, 90) if hovered else BG_CARD
    pygame.draw.rect(surface, col,    rect, border_radius=4)
    pygame.draw.rect(surface, BORDER, rect, width=1, border_radius=4)
    txt = font_small.render(label, True, TEXT_PRI)
    surface.blit(txt, txt.get_rect(center=(rect[0]+rect[2]//2, rect[1]+rect[3]//2)))

def draw_simulation(surface):
    bg = (10, 10, 20) if not env.is_day else (18, 22, 34)
    pygame.draw.rect(surface, bg,     (0, 0, SIM_W, HEIGHT))
    pygame.draw.rect(surface, BORDER, (0, 0, SIM_W, HEIGHT), width=1)

    for gx in range(40, SIM_W, 40):
        pygame.draw.line(surface, (30, 35, 55), (gx, 0), (gx, HEIGHT))
    for gy in range(40, HEIGHT, 40):
        pygame.draw.line(surface, (30, 35, 55), (0, gy), (SIM_W, gy))

    if len(trail) > 1:
        for i in range(1, len(trail)):
            r = i / len(trail)
            pygame.draw.line(surface, (int(167*r), int(139*r), int(250*r)), trail[i-1], trail[i], 1)

    for f in foods:
        pygame.draw.circle(surface, FOOD_COLOR, f, 5)
        pygame.draw.circle(surface, (0,0,0),    f, 5, width=1)

    col = agent_color()
    ax, ay = int(agent.x), int(agent.y)
    pygame.draw.circle(surface, col,      (ax, ay), 12)
    pygame.draw.circle(surface, TEXT_PRI, (ax, ay), 12, width=2)
    surface.blit(font_small.render("A", True, (255,255,255)), (ax-5, ay-7))

    state_labels = {"rest": "RIPOSO", "explore": "ESPLORAZIONE", "search_food": "RICERCA CIBO", "idle": "INATTIVO"}
    state_colors = {"rest": (96,165,250), "explore": (167,139,250), "search_food": (251,191,36), "idle": TEXT_SEC}
    st = font_small.render(f"● {state_labels.get(agent.state, agent.state.upper())}", True, state_colors.get(agent.state, TEXT_SEC))
    surface.blit(st, (10, HEIGHT - 26))

    ct = font_small.render("☀ DIURNO" if env.is_day else "☽ NOTTURNO", True, TEXT_SEC)
    surface.blit(ct, (SIM_W - ct.get_width() - 10, HEIGHT - 26))

def draw_panel(surface, mouse_pos):
    global buttons
    buttons = []
    px, y = PANEL_X, 10

    surface.blit(font_large.render("MONITOR CLINICO", True, TEXT_PRI), (px, y)); y += 28
    surface.blit(font_small.render(
        f"tick {tick:05d}  |  cibo {int(env.food_available):3d}  |  temp {env.temperature:.1f}°C",
        True, TEXT_SEC), (px, y)); y += 18

    # epsilon — quanto è ancora casuale la rete
    eps     = agent.brain.epsilon
    eps_col = GREEN if eps < 0.2 else YELLOW if eps < 0.6 else RED
    surface.blit(font_small.render(
        f"rete: epsilon {eps:.2f}  ({'esplora' if eps > 0.5 else 'apprende' if eps > 0.15 else 'esperto'})",
        True, eps_col), (px, y)); y += 20

    pygame.draw.line(surface, BORDER, (px, y), (px+PANEL_W, y)); y += 10

    # energia
    surface.blit(font_med.render("ENERGIA", True, TEXT_SEC), (px, y)); y += 18
    en_col = GREEN if agent.energy > 60 else YELLOW if agent.energy > 30 else RED
    draw_bar(surface, px, y, PANEL_W, 14, agent.energy, en_col)
    lbl = font_small.render(f"{int(agent.energy)}", True, TEXT_PRI)
    surface.blit(lbl, (px + PANEL_W//2 - lbl.get_width()//2, y)); y += 28

    pygame.draw.line(surface, BORDER, (px, y), (px+PANEL_W, y)); y += 10

    # ormoni
    surface.blit(font_med.render("ORMONI", True, TEXT_SEC), (px, y)); y += 20

    BTN_W, BTN_H, BAR_H, GAP = 28, 17, 8, 4

    for h in HORMONE_CFG:
        val = getattr(agent, h["key"])
        surface.blit(font_med.render(h["label"],  True, h["color"]),  (px, y))
        surface.blit(font_small.render(h["desc"], True, TEXT_SEC),     (px+112, y+2))
        vl = font_small.render(f"{int(val):3d}", True, TEXT_PRI)
        surface.blit(vl, (px+PANEL_W-vl.get_width(), y)); y += 17

        draw_bar(surface, px, y, PANEL_W, BAR_H, val, h["color"]); y += BAR_H + GAP

        total = len([None]*4) * (BTN_W+GAP) - GAP
        bx = px + (PANEL_W - total) // 2
        for lbl_txt, delta in [("−10",-10),("−1",-1),("+1",1),("+10",10)]:
            rect = pygame.Rect(bx, y, BTN_W, BTN_H)
            draw_button(surface, rect, lbl_txt, rect.collidepoint(mouse_pos))
            buttons.append({"rect": rect, "key": h["key"], "delta": delta})
            bx += BTN_W + GAP
        y += BTN_H + 8

    pygame.draw.line(surface, BORDER, (px, y), (px+PANEL_W, y)); y += 10

    # legenda
    surface.blit(font_med.render("COLORE AGENTE", True, TEXT_SEC), (px, y)); y += 18
    for name, desc, col in [("Rosso","stress/adrenalina",RED),("Verde","benessere",GREEN),("Blu","calma",(96,165,250))]:
        pygame.draw.circle(surface, col, (px+6, y+6), 5)
        surface.blit(font_small.render(f"{name}: {desc}", True, TEXT_SEC), (px+16, y)); y += 16


# ===== LOOP =====
running = True

while running:
    clock.tick(10)
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            agent.brain.save()          # salva pesi alla chiusura
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for btn in buttons:
                if btn["rect"].collidepoint(event.pos):
                    current = getattr(agent, btn["key"])
                    setattr(agent, btn["key"], clamp(current + btn["delta"], 0, 100))

    tick += 1
    env.update()
    spawn_food()
    agent.decide()
    agent.act(env)
    agent.update_hormones(env)

    # movimento
    dx, dy = 0, 0
    tx, ty, min_d = None, None, float("inf")
    for f in foods:
        d = distance(agent.x, agent.y, f[0], f[1])
        if d < min_d:
            min_d = d; tx, ty = f[0], f[1]

    if tx is not None:
        hunger = (100 - agent.energy) / 100
        dx += (tx - agent.x) * hunger * 0.08
        dy += (ty - agent.y) * hunger * 0.08

    dx += random.uniform(-1,1) * (agent.dopamine   / 100) * 4
    dy += random.uniform(-1,1) * (agent.dopamine   / 100) * 4
    dx += random.uniform(-1,1) * (agent.cortisol   / 100) * 3
    dy += random.uniform(-1,1) * (agent.cortisol   / 100) * 3
    dx += random.uniform(-1,1) * (agent.adrenaline / 100) * 5
    dy += random.uniform(-1,1) * (agent.adrenaline / 100) * 5

    speed = 3 * (1 - agent.melatonin / 130)
    agent.x = clamp(agent.x + dx * speed, 0, SIM_W)
    agent.y = clamp(agent.y + dy * speed, 0, HEIGHT)

    trail.append((int(agent.x), int(agent.y)))
    if len(trail) > 60:
        trail.pop(0)

    for f in foods[:]:
        if distance(agent.x, agent.y, f[0], f[1]) < 15:
            foods.remove(f)
            agent.energy    = clamp(agent.energy    + 25, 0, 100)
            agent.dopamine  = clamp(agent.dopamine  + 15, 0, 100)
            agent.serotonin = clamp(agent.serotonin +  8, 0, 100)

    screen.fill(BG)
    draw_simulation(screen)
    draw_panel(screen, mouse_pos)
    pygame.display.flip()

pygame.quit()