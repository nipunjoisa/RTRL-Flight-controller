"""Generate the System / Pipeline Architecture flowchart for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, PURPLE, TEAL, YELLOW, GREEN, PINK = (
    "#AED6F1", "#D2B4DE", "#A3E4D7", "#F9E79F", "#ABEBC6", "#F5B7B1")

fig, ax = plt.subplots(figsize=(7.4, 9.2), dpi=220)
ax.set_xlim(0, 100); ax.set_ylim(0, 132); ax.axis("off")

def box(cx, cy, w, h, text, color, fs=8.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=color, ec="#5D6D7E", lw=0.9))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="#1B2631", linespacing=1.45)
    return (cx, cy - h / 2, cy + h / 2)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color="#2C3E50",
                                 shrinkA=0, shrinkB=0))

W, H = 62, 9.5
rows = [
    ("JSBSim flight dynamics via jsbgym\nCessna 172 — AttitudeHoldTask", BLUE),
    ("Gymnasium wrapper stack\nNormalize → Fault → Wind → Trace", PURPLE),
    ("Tuned PID expert rollouts\n50 episodes × 300 steps → imitation dataset", TEAL),
    ("Imitation warm-start\n(offline BPTT, 20 epochs)", YELLOW),
    ("Warm-started weights\n(data/checkpoints/*.pt)", GREEN),
    ("ExperimentRunner — per step:\nact() → update() → log()", PINK),
]
y = 126.0
centres = []
for text, color in rows:
    centres.append(box(50, y, W, H, text, color))
    y -= 14.5

for i in range(len(centres) - 1):
    arrow(50, centres[i][1], 50, centres[i + 1][2])

# Branch row: the three controllers behind the one ABC
by = y + 1.0
bw, bh = 29, 11.0
cols = [
    (17, "PID baseline\nupdate() = no-op", BLUE),
    (50, "BPTT-LSTM baseline\nweights frozen", PURPLE),
    (83, "RTRL-RTU\nonline RTRL update", GREEN),
]
branch = [box(cx, by, bw, bh, t, c, fs=7.8) for cx, t, c in cols]
for cx, _, _ in cols:
    arrow(50, centres[-1][1], cx, by + bh / 2 + 0.6)

# Converge into tracing, then analysis
ty = by - 15.0
trace = box(50, ty, W, H, "TraceLogger → per-step parquet trace\n(one row per env step)", PINK)
for cx, _, _ in cols:
    arrow(cx, by - bh / 2, 50, ty + H / 2 + 0.6)

ay = ty - 14.5
box(50, ay, W, H, "Analysis: RMSE, jerk, time-to-recover\n→ results table and figures", GREEN)
arrow(50, trace[1], 50, ay + H / 2 + 0.6)

ax.set_title("System / Pipeline Architecture", fontsize=11, fontweight="bold",
             color="#1B2631", pad=6)
fig.savefig("assets/arch.png", bbox_inches="tight", facecolor="white")
print("wrote assets/arch.png")
