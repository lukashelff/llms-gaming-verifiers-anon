"""P0d - Panel (a) with reasoning models only: gpt-5 family (medium) plus Olmo-3.1-32B-Think,
gpt-oss-120b and gpt-oss-20b (medium effort). Same fixed canvas as P0c/P3c."""
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import fig1_fig4_common as c

c.style(base=15)
d5 = c.load(include_nano_family=False)
d5 = d5[d5.model.isin(["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.5-preview", "gpt-4o", "gpt-4-turbo"])][["model", "level", "completion_tokens", "outcome", "shortcut"]]

ipt = pd.read_csv(c.OSS_ROOT / "ipt_results/detailed_results.csv")
def load_oss(run, path, label, marker):
    outs = json.load(open(path))
    j = pd.DataFrame({"problem_id": [int(o["problem_id"]) for o in outs],
                      "completion_tokens": [o.get("completion_tokens") for o in outs],
                      "nofinal": [marker not in (o.get("model_completion") or "") for o in outs]})
    f_ = ipt[ipt.model_name == run][["problem_id", "isomorphic_correct", "is_reward_shortcut"]]
    g = j.merge(f_, on="problem_id", how="left", validate="1:1")
    g["level"] = (g.problem_id - 1) // 50 + 1
    g["shortcut"] = g.is_reward_shortcut.fillna(False).astype(bool) & ~g.nofinal
    g["solved"] = g.isomorphic_correct.fillna(False).astype(bool) & ~g.nofinal
    g["outcome"] = np.select([g.shortcut, g.solved], ["shortcut", "solved"], "failed")
    g["model"] = label
    g = g[~g.nofinal]
    return g[["model", "level", "completion_tokens", "outcome", "shortcut"]]

d = pd.concat([d5,
               load_oss("Olmo-3.1-32B-Think", c.OSS_ROOT / "olmo/Olmo-3.1-32B-Think/model_outputs.json", "Olmo-3.1-32B", "</think>"),
               load_oss("gpt-oss-120b-effort-medium", c.OSS_ROOT / "gpt-oss/gpt-oss-120b-effort-medium/model_outputs.json", "gpt-oss-120b", c.FINAL_MARKER)],
              ignore_index=True)
models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.5-preview", "gpt-4o", "gpt-4-turbo"]
GROUPS = {"RLVR": models[:3], "non-RLVR": models[3:]}
ymap = {m: (i if i < 3 else i + 0.55) for i, m in enumerate(models)}
fig, ax = plt.subplots(figsize=(5.2, 3.2), layout=None)
rng = np.random.default_rng(42)
for lo, hi in [(6, 10), (16, 20)]:
    ax.axvspan(lo - 0.5, hi + 0.5, color="#f4f4f4", zorder=0, lw=0)
x = d.level.to_numpy(float) + rng.uniform(-0.28, 0.28, len(d))
y = d.model.map(ymap).to_numpy(float)
k = (d.outcome == "solved").to_numpy(); c.strip(ax, x[k], y[k], c.BLUE, "up", rng, s=4, alpha=0.35)
k = (d.outcome == "shortcut").to_numpy(); c.strip(ax, x[k], y[k], c.RED, "down", rng, s=5, alpha=0.6, z=4)
for m in models:
    g = d[d.model == m]
    tok = g.completion_tokens.median(); n_sc = int(g.shortcut.sum())
    ax.text(1.03, ymap[m], f"{n_sc}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=11, color=c.RED if n_sc else "#444", clip_on=False)
top = -0.75
ax.text(1.03, top, "#\nsc.", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=10, color="#666", clip_on=False)
ax.set_yticks([ymap[m] for m in models]); ax.set_yticklabels(models, fontsize=11)
for grp, members in GROUPS.items():
    ys = [ymap[m] for m in members]
    ax.text(-0.50, np.mean(ys), grp, transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center", fontsize=12, clip_on=False)
    ax.plot([-0.46, -0.46], [ys[0] - 0.3, ys[-1] + 0.3], transform=ax.get_yaxis_transform(), color="#666", lw=0.8, clip_on=False)
ax.set_ylim(max(ymap.values()) + 0.5, -0.6); ax.set_xlim(0.4, 20.6); ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel("Task complexity (level)")
c.clean(ax)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c.BLUE, ms=5, alpha=0.7, label="Solved"),
                   Line2D([], [], marker="o", ls="", color=c.RED, ms=5, alpha=0.8, label="Shortcut")],
          loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=12, handletextpad=0.1, borderaxespad=0.1)
matplotlib.rcParams["savefig.bbox"] = None
fig.subplots_adjust(left=0.32, right=0.89, bottom=0.17, top=0.86)
for ext in (".pdf", ".png"):
    fig.savefig(c.OUT / ("p0g_level_strips_3plus3" + ext), bbox_inches=None, pad_inches=0)
print("saved", c.OUT / "p0g_level_strips_3plus3")
