"""Main-text OlmoLogic figure, two candidates.
A: full-width grouped bars of absolute scores (Think / OlmoLogic / Olmo-3.1 control) with OlmoLogic deltas annotated.
B: half-width paired horizontal delta bars (no side number columns)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import fig3_common as C

C.style()
V, _ = C.load_olmes()
TH, LG, O31 = "Olmo-3-7B-Think", "OlmoLogic-7B-Think", "Olmo-3.1-7B-Think"
GROUPS = [("Logical reasoning", [("SLR-Bench", "SLR-Bench"), ("LogiGLUE", "LogiGLUE"), ("Abduction", "Abduction"), ("bAbI", "bAbI"),
                                 ("NatLang", "LogiGLUE natlang"), ("CLUTRR", "CLUTTR"), ("ProntoQA", "LogiGLUE prontoqa"), ("RuleBERT", "RuleBERT"),
                                 ("KOR-Bench", "KOR-Bench"), ("ZebraLogic", "ZebraLogic")]),
          ("General reasoning & math", [("BBH", "BBH (27)"), ("AIME 2024", "AIME 2024"), ("AIME 2025", "AIME 2025"), ("MATH", "MATH"), ("GSM8K", "GSM8K")])]
COLS = {TH: "#9A9A9A", LG: C.COL[C.LOGIC], O31: "#C4B9E2"}
LAB = {TH: "Olmo-3-7B-Think", O31: "Olmo-3.1-7B-Think (ours)", LG: "OlmoLogic-7B-Think (ours)"}

# ---------------- A: grouped absolute bars, full width
fig = plt.figure(figsize=(5.5, 2.05))
ax = fig.add_axes([0.065, 0.21, 0.925, 0.63])
x = 0.0; xt, xl, gspan = [], [], []
w = 0.26
for g, items in GROUPS:
    g0 = x
    for lab, key in items:
        vals = [V[TH][key], V[LG][key], V[O31][key]]
        for k, (m, v) in enumerate(zip([TH, O31, LG], [vals[0], vals[2], vals[1]])):
            ax.bar(x + (k - 1) * w, v, width=w * 0.95, color=COLS[m], lw=0, zorder=3)
        d = V[LG][key] - V[TH][key]
        ax.text(x, max(vals) + 2.5, f"{d:+.1f}", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=C.dcolor(d))
        xt.append(x); xl.append(lab); x += 1.0
    gspan.append((g0 - 0.5, x - 0.5, g)); x += 0.55
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.2, rotation=28, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.6, x - 1.05); ax.set_ylim(0, 116); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Score (%)", labelpad=2)
ax.grid(axis="y", color=C.GRID, lw=0.5, zorder=0); C.despine(ax); ax.tick_params(axis="x", length=0)
for a, b, g in gspan:
    ax.plot([a + 0.1, b - 0.1], [112.5, 112.5], color=C.MUTED, lw=0.6, clip_on=False)
    ax.text((a + b) / 2, 114.2, g, ha="center", va="bottom", fontsize=6.2, style="italic", color=C.MUTED)
h = [Patch(facecolor=COLS[m], label=LAB[m]) for m in (TH, O31, LG)]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False, fontsize=6.0, handlelength=1.1, columnspacing=1.2, handletextpad=0.4)
C.save(fig, "main_bars")

# ---------------- B: paired horizontal delta bars, half width
fig = plt.figure(figsize=(2.75, 2.55))
ax = fig.add_axes([0.27, 0.13, 0.70, 0.74])
y = 0.0; yt, yl, gl = [], [], []
XMIN, XMAX = -8, 14
for g, items in GROUPS:
    gl.append((y, g)); y += 0.9
    for lab, key in items:
        dl, do = V[LG][key] - V[TH][key], V[O31][key] - V[TH][key]
        dlc, doc = np.clip(dl, XMIN, XMAX), np.clip(do, XMIN, XMAX)
        ax.barh(y - 0.19, dlc, height=0.36, color=C.COL[C.LOGIC], lw=0, zorder=3)
        ax.barh(y + 0.19, doc, height=0.36, color=C.OLMO31_7B, lw=0, zorder=3)
        ax.text((dlc + 0.35) if dl >= 0 else (dlc - 0.35), y - 0.19, f"{dl:+.1f}", va="center", ha="left" if dl >= 0 else "right", fontsize=5.6, color=C.dcolor(dl), fontweight="bold")
        yt.append(y); yl.append(lab); y += 1.0
    y += 0.3
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.2)
for gy, g in gl:
    ax.text(XMIN, gy + 0.32, g, fontsize=5.9, style="italic", color=C.MUTED, ha="left", va="center")
ax.set_ylim(y - 0.6, -0.4); ax.set_xlim(XMIN, XMAX + 1.2)
ax.axvline(0, color="#5a6470", lw=0.6, zorder=2); ax.grid(axis="x", color=C.GRID, lw=0.5, zorder=0)
ax.set_xticks([-5, 0, 5, 10]); C.despine(ax); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Δ vs. Olmo-3-7B-Think (points)", labelpad=1.5)
h = [Patch(facecolor=C.COL[C.LOGIC], label="OlmoLogic-7B-Think (ours)"), Patch(facecolor=C.OLMO31_7B, label="Olmo-3.1-7B-Think (ours, no SLR-Bench)")]
fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.02, 0.885), ncol=1, frameon=False, fontsize=6.0, handlelength=1.1, labelspacing=0.25, borderpad=0.0)
C.save(fig, "main_hbars")
