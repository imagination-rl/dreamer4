"""Build the shareable page of the instability audit (../page.html) from the data files and the figures.

Run plot.py first. The page reads only files of ../data and ../figures, so a rerun after more probes finish updates every number.
"""

import base64
import glob
import html
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
STAGE1 = ROOT.parent / "10-4-26-stage1-combinations" / "data" / "loop_metrics.csv"
PAIRS = ["combo_cevalue_warmup15", "combo_symexp_warmup15", "combo_symlog_warmup15", "combo_rw4_block2", "combo_rw4_depth5", "combo_rw4_fullres"]
LABEL = {"combo_cevalue_warmup15": "cevalue + warmup15", "combo_symexp_warmup15": "symexp + warmup15", "combo_symlog_warmup15": "symlog + warmup15",
         "combo_rw4_block2": "rw4 + block2", "combo_rw4_depth5": "rw4 + depth5", "combo_rw4_fullres": "rw4 + fullres",
         "loss_normalization": "base (grouped normalization)", "hl_gauss_value_loss": "cevalue (arm)", "norm_warmup15": "warmup15 (arm)",
         "symexp_hl_gauss_actor_critic": "symexp (arm)", "model_free": "model-free PPO"}


def group_of(run):
    return run[len("a2_grouped_"):].rsplit("_s", 1)[0]


def fmt(x, digits = 2, sign = False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{digits}f}" if sign else f"{x:,.{digits}f}"


def table(header, rows, note = None, cls = ""):
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    caption = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="tablewrap {cls}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{caption}'


def figure(name, caption):
    path = FIG / name
    if not path.exists():
        return f'<p class="pending">Figure {name} appears when its probe finishes.</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><div class="plate"><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"></div><figcaption>{caption}</figcaption></figure>'


# ---------------------------------------------------------------- data

def load():
    out = {}
    replay = pd.read_csv(DATA / "replay_loops.csv")
    replay["group"] = replay.run.map(group_of)
    out["replay"] = replay
    metrics = pd.read_csv(STAGE1)
    out["metrics"] = metrics
    out["by_return"] = pd.read_csv(DATA / "compare_entropy_by_return.csv")
    out["by_loop"] = pd.read_csv(DATA / "compare_entropy_by_loop.csv")
    out["ppo_step"] = pd.read_csv(DATA / "compare_ppo_step.csv")
    out["onsets"] = pd.read_csv(DATA / "flip_onsets.csv")
    out["bursts"] = pd.read_csv(DATA / "flip_bursts_summary.csv")
    out["dispersion"] = pd.read_csv(DATA / "dispersion.csv")
    files = glob.glob(str(DATA / "probe" / "*__loop_*.csv"))
    out["probe"] = pd.concat([pd.read_csv(f) for f in files]) if files else None
    files = glob.glob(str(DATA / "counterfactual" / "*__loop_*.csv"))
    out["cf"] = pd.concat([pd.read_csv(f) for f in files]) if files else None
    return out


def flip_section(d):
    late = d["replay"][(d["replay"].loop >= 100) & d["replay"].group.isin(PAIRS)]
    rows = []
    for group in PAIRS:
        runs = []
        for _, g in late[late.group == group].groupby("run"):
            r2 = np.corrcoef(g.flipped_episode_frac, g.return_mean)[0, 1] ** 2 if g.flipped_episode_frac.std() > 0 else 0.
            noise = lambda x: (np.asarray(x)[4:-4] - np.convolve(x, np.ones(9) / 9, "valid")).std() / max(np.mean(x), 1.)
            runs.append({"flip": g.flipped_episode_frac.mean(), "ret": g.return_mean.mean(), "up": g.return_upright_eps.mean(), "r2": r2,
                         "noise_all": noise(g.return_mean.to_numpy()), "noise_up": noise(g.return_upright_eps.ffill().to_numpy())})
        m = pd.DataFrame(runs).mean()
        rows.append([LABEL[group], f"{100 * m.flip:.0f} %", fmt(m.ret, 0), fmt(m.up, 0), fmt(m.up - m.ret, 0), fmt(m.r2), fmt(m.noise_all, 3), fmt(m.noise_up, 3)])
    onsets = d["onsets"]
    return rows, len(onsets), int(onsets.recovered.sum())


def entropy_tables(d):
    br = d["by_return"].groupby(["group", "R"]).agg(entropy = ("real_entropy", "mean"), loop = ("loop", "mean"), n = ("run", "count")).reset_index()
    order = PAIRS + ["loss_normalization", "hl_gauss_value_loss", "norm_warmup15", "symexp_hl_gauss_actor_critic", "model_free"]
    rows = []
    for group in order:
        cells = [LABEL[group]]
        for R in (1000, 2000, 3000):
            hit = br[(br.group == group) & (br.R == R)]
            cells.append(f"{fmt(hit.entropy.iloc[0], 1, True)} <span class='dim'>(loop {hit.loop.iloc[0]:.0f})</span>" if len(hit) else "not reached")
        rows.append(cells)
    bl = d["by_loop"]
    rows2 = []
    for group in PAIRS + ["model_free"]:
        cells = [LABEL[group]]
        for window in ("50-74", "100-124", "150-174"):
            hit = bl[(bl.group == group) & (bl.window == window)]
            cells.append(f"{fmt(hit.real_entropy.iloc[0], 1, True)} <span class='dim'>(return {hit['return'].iloc[0]:,.0f})</span>" if len(hit) else "–")
        rows2.append(cells)
    return rows, rows2


def gap_table(d):
    m = d["metrics"]
    e = m[m.tag == "cleanrl_updated_ppo/entropy"][["run", "loop", "value"]].copy()
    e["loop"] += 1
    x = d["replay"].merge(e, on = ["run", "loop"])
    x = x[(x.loop >= 100) & x.group.isin(PAIRS)]
    closed = m[m.tag.isin(["audit/closed_loop/dream_flipped_at_end_frac", "audit/closed_loop/true_flipped_at_end_frac"]) & (m.loop >= 100)].copy()
    closed["group"] = closed.run.map(group_of)
    closed = closed.pivot_table(index = "group", columns = "tag", values = "value")
    rows = []
    for group in PAIRS:
        g = x[x.group == group]
        rows.append([LABEL[group], fmt(g.value.mean(), 1, True), fmt(g.real_entropy.mean(), 1, True), fmt(g.real_entropy_upright.mean(), 1, True),
                     fmt(g.real_entropy_flipped.mean(), 1, True), fmt((g.value - g.real_entropy_upright).mean(), 2, True),
                     f"{100 * closed.loc[group, 'audit/closed_loop/dream_flipped_at_end_frac']:.0f} %",
                     f"{100 * closed.loc[group, 'audit/closed_loop/true_flipped_at_end_frac']:.0f} %"])
    return rows


def ppo_tables(d):
    s = d["ppo_step"]
    rows = []
    for group in PAIRS + ["loss_normalization", "model_free"]:
        cells = [LABEL[group]]
        for window in ("25-49", "50-99", "100-174"):
            hit = s[(s.group == group) & (s.window == window)]
            cells.append(f"{fmt(hit.approx_kl.iloc[0], 3)} / {100 * hit.clip_fraction.iloc[0]:.0f} %" if len(hit) and np.isfinite(hit.approx_kl.iloc[0]) else "–")
        rows.append(cells)
    names = {"ppo_kl": "PPO update size (approx-KL)", "clip_frac": "PPO clip fraction", "dream_entropy": "entropy on dream states",
             "real_entropy_upright": "entropy on upright real states", "x_velocity": "forward speed", "saturation": "actions at the limits",
             "flow_loss": "world-model flow loss", "reward_mae": "world-model reward error", "critic_ev": "critic explained variance",
             "dream_return": "dream return", "wm_grad_norm": "world-model gradient norm", "return_mean": "real return"}
    b = d["bursts"]
    rows2 = [[names.get(r.measure, r.measure), fmt(r.change_before_burst, 3, True), fmt(r.change_all_loops, 3, True), fmt(r.p_burst_vs_all, 2), fmt(r.lag1_rho, 3, True)]
             for r in b.itertuples()]
    return rows, rows2


def drift_tables(d):
    p = d["probe"]
    if p is None:
        return None
    p = p.copy()
    p["group"] = p.run.map(group_of)
    metrics = d["metrics"]
    kl = metrics[metrics.tag == "cleanrl_updated_ppo/approx_kl"]
    loop_of = {"loop_50.pt": 49, "loop_100.pt": 99, "loop_175.pt": 174}
    rows, per_ckpt = [], []
    for (run, ckpt), g in p[p.t >= 100].groupby(["run", "checkpoint"]):
        get = lambda m, col = "upright_mean": g[g.measure == m][col].mean() if (g.measure == m).any() else np.nan
        per_ckpt.append({"group": group_of(run), "checkpoint": ckpt, "drift": get("drift_kl_full"), "drift_w40": get("drift_kl_w40"),
                         "drift_median": get("drift_kl_full", "median"),
                         "context_w40": get("context_kl_w40", "mean"), "context_w8": get("context_kl_w8", "mean"), "context_w200": get("context_kl_w200", "mean"),
                         "entropy": get("entropy_full"),
                         "ppo": kl[(kl.run == run) & kl.loop.between(loop_of[ckpt] - 4, loop_of[ckpt])].value.mean()})
    c = pd.DataFrame(per_ckpt)
    order = [(g, ck) for ck in ("loop_50.pt", "loop_100.pt", "loop_175.pt") for g in ["loss_normalization", "hl_gauss_value_loss", "norm_warmup15", "symexp_hl_gauss_actor_critic"] + PAIRS]
    for group, ckpt in order:
        g = c[(c.group == group) & (c.checkpoint == ckpt)]
        if not len(g):
            continue
        ratio = (g.drift / g.ppo).mean()
        rows.append([LABEL[group], ckpt[5:-3], str(len(g)), fmt(g.entropy.mean(), 1, True), fmt(g.ppo.mean(), 3), fmt(g.drift.mean(), 3),
                     fmt(g.drift_median.mean(), 3), fmt(g.drift_w40.mean(), 3), f"{ratio:.1f}×"])
    summary = {"n": len(c), "ratio_median": float((c.drift / c.ppo).median()), "ratio_min": float((c.drift / c.ppo).min()),
               "ratio_max": float((c.drift / c.ppo).max()), "frac_above": float((c.drift > c.ppo).mean()),
               "context_w40_max": float(c.context_w40.max()), "context_w200_max": float(c.context_w200.max()) if c.context_w200.notna().any() else np.nan,
               "context_w8_max": float(c.context_w8.max()) if c.context_w8.notna().any() else np.nan,
               "context_default_max": float(c[~c.group.str.startswith("combo_rw4")].context_w40.max()),
               "context_default_median": float(c[~c.group.str.startswith("combo_rw4")].context_w40.median()),
               "context_rw4_min": float(c[c.group.str.startswith("combo_rw4")].context_w40.min()) if c.group.str.startswith("combo_rw4").any() else np.nan,
               "context_rw4_max": float(c[c.group.str.startswith("combo_rw4")].context_w40.max()) if c.group.str.startswith("combo_rw4").any() else np.nan,
               "drift_rw4_min": float(c[c.group.str.startswith("combo_rw4")].drift.min()) if c.group.str.startswith("combo_rw4").any() else np.nan}
    return rows, summary


def counterfactual_tables(d):
    cf = d["cf"]
    if cf is None:
        return None
    branches = cf[~cf["mode"].isin(["entropy"])]
    key = ["run", "loop", "episode"]
    onsets = branches.groupby(key).ngroups
    rows = []
    for offset in sorted(branches.offset.unique(), reverse = True):
        b = branches[branches.offset == offset]
        get = lambda mode: b[b["mode"] == mode].flip_frac
        rows.append([f"{offset} steps before", f"{100 * get('recorded').mean():.0f} %", f"{100 * get('sample').mean():.0f} %", f"{100 * get('mean').mean():.0f} %"])
    at30 = branches[(branches.offset == 30)].pivot_table(index = key, columns = "mode", values = "flip_frac")
    at30 = at30[at30.recorded > 0.5]  # the restore reproduced the flip
    classes = {
        "bad luck: the mean action stays upright and at most 4 of 16 new samples flip": ((at30["mean"] < 0.5) & (at30["sample"] <= 0.25)).sum(),
        "mixed: 5 to 11 of 16 new samples flip": ((at30["sample"] > 0.25) & (at30["sample"] < 0.75)).sum(),
        "committed: at least 12 of 16 new samples flip": (at30["sample"] >= 0.75).sum(),
        "the mean action flips, but at most 4 of 16 samples do": ((at30["mean"] >= 0.5) & (at30["sample"] <= 0.25)).sum(),
    }
    numbers = {"sample_30": 100 * at30["sample"].mean(), "mean_30": 100 * at30["mean"].mean()}
    ent = cf[cf["mode"] == "entropy"]
    on, up = ent[ent.kind == "onset"], ent[ent.kind == "upright"]
    entropy_rows = [
        ["30 steps before a flip", fmt(on.entropy_pre.mean(), 2, True), fmt(on.surprise_pre.mean(), 2, True), fmt((on.surprise_pre - on.entropy_pre).mean(), 2, True)],
        ["the same episodes, 150-250 steps before", fmt(on.entropy_base.mean(), 2, True), fmt(on.surprise_base.mean(), 2, True), fmt((on.surprise_base - on.entropy_base).mean(), 2, True)],
        ["upright episodes, steps 670-699", fmt(up.entropy_pre.mean(), 2, True), fmt(up.surprise_pre.mean(), 2, True), fmt((up.surprise_pre - up.entropy_pre).mean(), 2, True)],
    ]
    numbers["entropy_rise"] = float((on.entropy_pre - on.entropy_base).mean())
    return rows, onsets, len(at30), classes, entropy_rows, numbers


# ---------------------------------------------------------------- page

CSS = """
:root {
  /* Layout: one reading column, wide figure plates, tables that scroll on their own. */
  --bg: #f6f7f5; --surface: #ffffff; --fg: #1d2421; --muted: #5b6661; --rule: #d9dedb; --accent: #1f6f5c; --warn: #a34a1e; --plate: #ffffff;
  --display: "Fraunces", Georgia, serif; --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif; --mono: "JetBrains Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141917; --surface: #1c2320; --fg: #e3e8e5; --muted: #9aa6a0; --rule: #33403a; --accent: #6cc4a8; --warn: #e59466; --plate: #f4f5f3; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141917; --surface: #1c2320; --fg: #e3e8e5; --muted: #9aa6a0; --rule: #33403a; --accent: #6cc4a8; --warn: #e59466; --plate: #f4f5f3; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 17px; line-height: 1.6; }
main { max-width: 900px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 80px; }
h1, h2, h3 { font-family: var(--display); line-height: 1.2; text-wrap: balance; }
h1 { font-size: 2.3rem; margin: 0 0 6px; font-weight: 600; }
h2 { font-size: 1.55rem; margin: 52px 0 10px; padding-top: 18px; border-top: 1px solid var(--rule); font-weight: 600; }
h3 { font-size: 1.15rem; margin: 28px 0 6px; }
p, li { max-width: 68ch; }
.lede { color: var(--muted); font-size: 1.05rem; }
.eyebrow { font-family: var(--mono); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }
.verdicts { display: grid; gap: 10px; margin: 18px 0; }
.verdict { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 16px; display: grid; grid-template-columns: 9.5rem 1fr; gap: 14px; }
.verdict .tag { font-family: var(--mono); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--accent); padding-top: 3px; }
.verdict .tag.no { color: var(--warn); }
.verdict p { margin: 0; }
@media (max-width: 560px) { .verdict { grid-template-columns: 1fr; gap: 2px; } }
.tablewrap { overflow-x: auto; margin: 14px 0 4px; }
table { border-collapse: collapse; font-size: 0.88rem; font-variant-numeric: tabular-nums; min-width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 0.8rem; }
td:first-child { white-space: nowrap; }
.dim { color: var(--muted); font-size: 0.85em; }
.note { color: var(--muted); font-size: 0.86rem; margin-top: 4px; }
.pending { color: var(--warn); font-style: italic; }
figure { margin: 22px 0; }
.plate { background: var(--plate); border-radius: 6px; padding: 8px; overflow-x: auto; }
.plate img { display: block; width: 100%; height: auto; min-width: 560px; }
figcaption { color: var(--muted); font-size: 0.88rem; margin-top: 6px; max-width: 75ch; }
.box { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 6px 18px; margin: 16px 0; }
code { font-family: var(--mono); font-size: 0.86em; }
a { color: var(--accent); }
"""


def build():
    d = load()
    flip_rows, n_onsets, n_recovered = flip_section(d)
    ent_return_rows, ent_loop_rows = entropy_tables(d)
    gap_rows = gap_table(d)
    ppo_rows, burst_rows = ppo_tables(d)
    drift = drift_tables(d)
    cf = counterfactual_tables(d)
    disp = d["dispersion"]
    mf_kl = d["ppo_step"][d["ppo_step"].group == "model_free"].approx_kl.mean()

    if drift:
        drift_rows, ds = drift
        drift_html = table(["setting", "checkpoint (loop)", "runs", "policy entropy", "PPO step (logged KL)", "world-model drift (KL, mean)", "drift, median state", "drift, 40-frame context", "drift ÷ PPO step"],
                           drift_rows, note = "PPO step: the logged approx-KL of the last 5 loops before the checkpoint (dream states). World-model drift: the probe KL on upright real states at steps 100-999 of 32 real episodes. Both are KL(before ‖ after) of the same policy, so they are in the same units. The mean is about twice the median: most states move a little and some move a lot.")
        drift_text = (f"<p>Over all {ds["n"]} checkpoints, one loop of world-model training moves the acting policy by a median of "
                      f"<b>{ds['ratio_median']:.1f} times</b> the PPO step of the same loop (range {ds['ratio_min']:.1f}× to {ds['ratio_max']:.1f}×). "
                      f"The drift is larger than the PPO step in {100 * ds['frac_above']:.0f} % of the checkpoints. "
                      f"For comparison, one model-free PPO update has an approx-KL of about {mf_kl:.3f}.</p>")
        context_text = (f"<p>The same probe also tests a second suspect: the context length. In the real env the policy acts with the cache of up to 1,000 past frames; "
                        f"in training it never sees more than 40. For the base, the arms and the value pairs (the default attention), cutting the real context to the last 40 frames "
                        f"changes the policy very little in a typical checkpoint (median of the checkpoint means: KL {ds['context_default_median']:.3f}). A few states change more: "
                        f"the largest checkpoint mean is {ds['context_default_max']:.2f} (base, seed 42, loop 100), with a median state of 0.01. So there the long context is a minor, "
                        f"occasional effect, far below the drift. "
                        f"For the three rw4 pairs, which change the attention (block2, depth5 without the attention pool, fullres), the change is larger: "
                        f"a mean KL of {fmt(ds['context_rw4_min'], 3)} to {fmt(ds['context_rw4_max'], 3)} per checkpoint. That is a second, smaller mismatch in those variants; "
                        f"their world-model drift (from {fmt(ds['drift_rw4_min'], 3)}) is still larger.</p>")
    else:
        drift_html = '<p class="pending">The drift probe is still running.</p>'
        drift_text = context_text = ""

    if cf:
        cf_rows, cf_onsets, cf_valid, cf_classes, cf_entropy_rows, cf_numbers = cf
        cf_html = (table(["branch point", "recorded actions (check)", "16 new samples of the policy", "mean action (no noise)"], cf_rows,
                         note = f"{cf_onsets} recorded flip onsets. Each cell: the fraction of branches that end flipped 60 steps after the recorded onset.")
                   + f"<p>At 30 steps before the onset, the restore reproduced the flip in {cf_valid} onsets. Those onsets fall into these groups:</p>"
                   + table(["kind of flip", "onsets"], [[k, str(v)] for k, v in cf_classes.items()])
                   + "<h3>Why the actions look surprising just before a flip</h3>"
                   + table(["window", "policy entropy", "surprise of the taken action", "surprise − entropy"], cf_entropy_rows,
                           note = "Entropy: how spread the policy is at that state. Surprise: −log π(a|s) of the action that was taken. If an unlucky draw caused the flip, the surprise would rise above the entropy. If the policy itself became unsure, the entropy rises and the surprise follows it."))
    else:
        cf_html = '<p class="pending">The counterfactual probe is still running.</p>'

    if cf:
        luck = list(cf_classes.values())[0]
        cf_verdict = (f'<div class="verdict"><span class="tag">New</span><p><b>Flips start as bad luck.</b> From the exact state 30 steps before {cf_valid} recorded flips, '
                      f'new samples of the same policy flip only {cf_numbers['sample_30']:.0f} % of the time and its mean action {cf_numbers['mean_30']:.0f} %. '
                      f'{luck} of {cf_valid} flips were a rare draw of the exploration noise. As the body tips, the policy becomes unsure (its entropy rises by {cf_numbers['entropy_rise']:.1f} nats), '
                      f'because the dreams never show these states.</p></div>')
    else:
        cf_verdict = ""
    disp_text = (f"median {disp.dispersion.median():.1f}; {int((disp.p_chance < 0.05).sum())} of {len(disp)} runs above chance at p &lt; 0.05; "
                 f"median lag-1 autocorrelation {disp.lag1_autocorr.median():+.2f}")

    body = f"""
<main>
<p class="eyebrow">Dreamer 4 on HalfCheetah · stage-1 runs · audit of 2026-10-04</p>
<h1>HalfCheetah Instability Audit</h1>
<p class="lede">Why do the imagination-trained runs drop in return late in training? This page tests four explanations, two from each side of a
conversation about run <code>a2_grouped_combo_cevalue_warmup15</code>. Every number comes from the training logs, the replay buffers, or probes of the saved checkpoints.</p>

<div class="verdicts">
  <div class="verdict"><span class="tag">Mostly</span><p><b>The drops are flips.</b> The cheetah turns onto its back and never recovers. Flipped episodes explain about 40 % of the loop-to-loop
  return variance; without them the loop-to-loop return is 1.2-2.6 times smoother.</p></div>
  {cf_verdict}
  <div class="verdict"><span class="tag">Supported</span><p><b>Moving representations.</b> One loop of world-model training moves the policy as much as the PPO update does, or more, and nothing limits that change.</p></div>
  <div class="verdict"><span class="tag">Partly</span><p><b>The policy learns too fast.</b> Each PPO update moves the policy 4-7 times further than in model-free PPO with the same settings. But no single large update comes before a flip burst.</p></div>
  <div class="verdict"><span class="tag">Partly</span><p><b>Entropy collapse is normal for a Beta policy.</b> True for its speed per env step. But for the return it buys, the policy is 2-3.5 nats narrower than model-free PPO.</p></div>
  <div class="verdict"><span class="tag no">Not supported</span><p><b>Confident in imagination, normal in reality.</b> On the same kind of states (upright) the policy is equally confident in both. Its only
  unsure states are flipped ones, which the dreams never show.</p></div>
</div>

<h2>How to read the numbers</h2>
<p>The policy outputs, for each of the 6 joints, a Beta distribution over the action range −1 to +1. Two quantities describe it.</p>
<p><b>Entropy</b> measures how spread out the actions are. It is in <b>nats</b> (natural-log units). The logs report the sum over the 6 joints,
on the −1 to +1 scale. Useful anchors:</p>
<ul>
<li><b>+4.16</b>: the uniform policy: any action is as likely as any other (6 × log 2). Training starts near here.</li>
<li><b>0</b>: each joint's actions spread over a band about 1.0 wide (half of the full range).</li>
<li><b>−3</b>: a band about 0.6 wide; <b>−6</b>: about 0.37 wide. Each −4.16 nats (6 × log 2) halves the band of every joint.</li>
</ul>
<p>The rule of thumb: the band width of one joint is about e<sup>H/6</sup> on the −1 to +1 scale. Negative entropy is normal for a continuous policy; it only means a narrow band.</p>
<p><b>KL divergence</b> measures how far the policy moved between two versions, also in nats. If only one joint moves, KL 0.05 means its preferred
action moved by about a third of its own band; KL 0.2 by about two thirds. PPO's <b>approx-KL</b> is an estimate of this for one update.
The <b>clip fraction</b> is the fraction of training samples whose probability changed by more than 20 % in the update. PPO stops pushing on those.</p>
<p><b>Surprise</b> is −log π(a|s) of one action that was actually taken. Its average over many states equals the entropy, so the replay's stored
log-probabilities give the entropy on the real states of every loop, after the fact.</p>
{figure("primer.png", "Left: one joint's action distribution at five entropy levels (the total assumes all 6 joints look the same). Right: the policy before an update (black) and after two updates of different KL.")}

<h2>The setup in one paragraph</h2>
<p>One <b>loop</b> runs 64 real episodes of 1,000 steps (64,000 env steps). It then trains the world model for 13 steps, and then does one PPO update on dreams.
A dream starts from 8 real frames of the replay and imagines 32 more steps. The update uses 4 × 512 dreams × 32 steps (65,536 transitions), 10 epochs of 32 minibatches,
clip 0.2, learning rate 3e-4 decayed linearly over 500 loops, and entropy weight 0. The policy head is a 2-layer network that reads the world model's 128-dim agent token,
so it shares its input with the world model.</p>
<p>The <b>model-free reference</b> (<code>sm_cleanrl_updated_PPO</code>, 2 seeds) uses the same Beta policy and the same PPO settings: 64,000 real steps per iteration, 10 × 32 minibatches,
clip 0.2, learning rate 3e-4 with the same decay, and entropy weight 0. Its actor is a separate 64-64 network on the raw state. One loop and one model-free iteration therefore use the
same env steps and the same number of gradient steps.</p>

<h2>What the instability is: flips</h2>
<p>An episode is <b>flipped</b> when the cheetah's body angle passes 90°, so it moves on its back. In the stage-1 pair runs (loops 100-174):</p>
{table(["pair (3 seeds)", "episodes flipped", "return, all", "return, upright episodes", "cost of flips", "variance explained by flips (R²)", "noise, all", "noise, upright only"], flip_rows,
       note = "Noise: the loop-to-loop scatter of the return around its 9-loop mean, divided by the mean. Upright episodes: fewer than 10 % flipped steps.")}
<p>Of {n_onsets:,} flip onsets in the replays, <b>{n_recovered} recovered</b>. A flip is permanent for the rest of the episode.</p>
{figure("flipped_states.png", "Left: policy entropy on upright and on flipped real states. On flipped states the policy is close to uniform. Right: return noise with and without the flipped episodes.")}
<p>The left panel matters for everything below. On flipped states the policy entropy is 2.5 to 4.2 nats, close to the uniform 4.16: the policy has almost no training there.
The world model rarely dreams a flip (next section), so PPO almost never sees flipped states.</p>

<h2>Question 1: Is the entropy collapse normal?</h2>
<p><i>The other researcher: "Entropy is expected to drop pretty logarithmic-like with a Beta distribution as it concentrates. Check it against the non-world-model variant."</i></p>
{figure("entropy.png", "Policy entropy on real states. Left: against env steps (loops). Right: against the real return. Dashed: model-free PPO with the same settings.")}
<p>Against env steps (left), the world-model policies narrow <b>more slowly</b> than model-free PPO. So the speed of the collapse is not unusual.</p>
<p>Against the return (right), the picture flips. To reach the same real return, the world-model policies are much narrower:</p>
{table(["setting", "entropy at return 1,000", "at return 2,000", "at return 3,000"], ent_return_rows,
       note = "Real-state entropy at the first loop at which the 5-loop mean return reaches the value (mean of the seeds that reach it).")}
<p>At return 3,000 the model-free policy still spreads each joint over a band about 1.2 wide (+0.95 nats). The best world-model pairs are at −2.1 to −2.7 nats, a band about 0.65 wide.
The world-model policy commits to a narrow gait that earns about half of what the same narrowness earns model-free. The dreams reward that gait, but the real env does not reward it as much.</p>

<h2>Question 2: Confident in imagination, normal in reality?</h2>
<p><i>"It may be an imagination thing. Is it highly confident in imagination rollouts and normal in real rollouts?"</i></p>
{table(["pair", "dream entropy", "real entropy, all", "real, upright", "real, flipped", "dream − real upright", "dreams that end flipped", "real rollouts that end flipped"], gap_rows,
       note = "Loops 100-174. Dream entropy: the logged entropy of the PPO minibatch. Real entropy: from the replay log-probs of the next loop, which the same policy collected. Last two columns: the in-run audit runs a dream and the real env from the same start state and the same policy.")}
<p>On all real states, the value pairs look more confident in dreams (dream entropy below real). That difference comes only from the flipped real states, where the policy is near-uniform.
On upright states, dream and real entropy agree within about 0.15 nats for the value pairs. The rw4 pairs are even <i>less</i> confident in dreams.
The real problem is in the last two columns: from the same start states, 9-41 % of real rollouts end flipped against 3-8 % of dreams (25-41 % in the value pairs). The dreams leave out the flip, so the policy never learns to avoid it.</p>

<h2>Question 3: Moving representations</h2>
<p><i>"My guess would be representation quality, or moving representations from the world model."</i></p>
<p>The policy head reads the world model's agent token. Each loop trains the world model first, and that changes the token, and therefore the policy, before PPO runs. PPO's clipping
only limits PPO's own change; nothing limits this one.</p>
<p><b>Probe.</b> Load a checkpoint, run one loop of world-model training (13 steps, batch 128 × 200 frames, the run's own optimizer state) on its replay, and keep the policy head fixed.
Then compare the acting policy before and after on 32 real episodes. The probe replays the real episodes exactly as the env loop did; on a checkpoint's own episodes it reproduces the stored log-probabilities to within 0.01 nats.</p>
{drift_text}
{drift_html}
{figure("policy_change.png", "How far the acting policy moves in one loop: the PPO update (blue) and the world-model drift (red), with the model-free PPO update for scale (dashed).")}
{context_text}

<h2>Question 4: Does the policy learn too fast?</h2>
<p><i>"I think the policy head is learning too fast, leading to more exploitation in the second half."</i></p>
{table(["setting", "loops 25-49", "loops 50-99", "loops 100-174"], ppo_rows,
       note = "Each cell: approx-KL of one PPO update / clip fraction. Same learning rate, epochs and minibatches in all rows.")}
<p>With identical PPO settings, each world-model PPO update moves the policy 4-7 times further than a model-free update (approx-KL 0.03-0.05 against about {mf_kl:.3f}),
and 30 % of the samples hit the clip, against 6 %. Add the world-model drift above, and the policy moves many times faster per env step than model-free PPO.</p>
<p>But no single fast update comes before a drop. A flip burst is a loop with at least 15 % flipped episodes after 5 calm loops (24 bursts in 13 runs). The loops before a burst look like any other loop:</p>
{table(["measure", "change before a burst", "change in any loop", "p (burst vs any)", "lag-1 correlation with flips"], burst_rows,
       note = "Change: the mean of loops L−3..L−1 minus the mean of loops L−10..L−6. Lag-1 correlation: within-run Spearman ρ between a measure in loop L−1 and the flipped fraction in loop L, both detrended (5,458 loop pairs).")}
<p>The number of flipped episodes per loop varies more than chance would give at a slowly changing rate (binomial dispersion test: {disp_text}).
So each loop's policy has its own flip risk, but a high-risk loop is not followed by another one. Each update re-draws the risk at random.
This fits a policy that moves a lot every loop, not one bad update.</p>

<h2>Luck or gait? Replaying the flips in the simulator</h2>
<p>The HalfCheetah observation holds the full simulator state except the x position, which has no effect on the dynamics. So the probe restores the exact state at 100, 30 or 10 steps
before a recorded flip. From there it runs (a) the recorded actions, to check the restore, (b) 16 new samples of the same policy, and (c) the policy's mean action with no noise.</p>
{cf_html}
{figure("counterfactual.png", "Fraction of branches that end flipped. Grey: recorded actions; red: new samples; blue: mean action.")}

<h2>What this means</h2>
<p>Put together, the evidence gives one chain of events:</p>
<ol>
<li><b>The world model rarely shows the flip.</b> From the same states, dreams end flipped 3-8 % of the time and reality 9-41 %. So PPO almost never trains on tipping or flipped states,
and the policy there is close to random.</li>
<li><b>A flip starts with a rare unlucky draw of the action noise.</b> From 30 steps before a recorded flip, the same policy usually stays upright. Once the body tips, the policy
is on states it never trained on: its entropy rises and it cannot recover. A flip is then permanent for the episode.</li>
<li><b>The flip risk changes every loop, at random.</b> The policy moves a lot in each loop: the PPO update is 4-7 times the model-free size, and the world-model drift adds
several times more, with nothing to limit it. Each change re-draws how often the gait meets the unlucky states. That is why the drops look random and do not repeat.</li>
<li><b>The policy is narrow for the return it gets.</b> It commits to gaits that the dreams overrate, and a narrow, fast gait spends more time near the edge.</li>
</ol>
<p>So the answer to "settings or luck?" is both: the settings set the flip risk and how fast it moves, and the noise decides which episodes and loops flip.</p>
<h3>Tests that follow from this</h3>
<ul>
<li><b>Limit the drift.</b> The code already has <code>preserve_policy_during_world_model_updates</code> (a KL penalty on the policy during world-model training) and
<code>reanchor_heads_after_world_model_updates</code> (refit the heads after each world-model phase). Neither was tested on the current base.</li>
<li><b>Smaller PPO steps.</b> The queued <code>lr150</code> runs halve the step. A target-KL early stop (CleanRL's <code>target_kl</code>) would cap it directly.</li>
<li><b>Slower world model.</b> The queued <code>wmlr300</code> runs halve the world-model learning rate and so the drift.</li>
<li><b>Entropy bonus, with care.</b> The queued <code>ent001</code> and <code>ent01</code> runs widen the policy. That can help the narrow-gait problem, but more action noise can also mean more
unlucky draws. Watch their flipped fraction, not only the return.</li>
<li><b>Teach the world model the flip.</b> Oversample the replay windows that contain a flip onset in world-model training, so that dreams can flip and PPO learns to avoid and recover.</li>
<li><b>Measure the mean-action return.</b> The mean action flipped far less often in the counterfactual. An evaluation with the mean action separates the gait's quality from the noise.</li>
</ul>

<h2>Method notes and limits</h2>
<ul>
<li>Runs: the 18 stage-1 pair runs (175 loops, seeds 4, 42, 1), the base (4 seeds) and the arms (3 seeds, 100 loops). Model-free: 2 seeds.</li>
<li>Real-state entropy is the mean surprise of the recorded actions. It is exact on average, and it needs no model.</li>
<li>The dream entropy is logged after the update on the last minibatch. The real entropy belongs to the next loop's episodes, which that policy collected.</li>
<li>The drift probe runs on the CPU, with gradient accumulation in micro-batches of 8 (the run used 32): the same gradient; the loss-normalization statistic sees 8 rows instead of 32.
Its world-model loss matches the logged loss of the run (9.83-9.94 against 9.95-9.99 at loop 50 of the base, seed 4).</li>
<li>The loop-175 checkpoints did not collect episodes. The drift and counterfactual probes use the episodes of loops 170-174, which a policy a few updates older collected.</li>
<li>The model-free comparison has only 2 seeds and no flip measure.</li>
<li>Scripts and data: <code>report/10-4-26-instability-audit/</code> in the repository.</li>
</ul>
</main>
"""
    page = f"""<title>HalfCheetah Instability Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=JetBrains+Mono:wght@400&family=Source+Sans+3:wght@400;600&display=swap">
<style>{CSS}</style>
{body}"""
    (ROOT / "page.html").write_text(page)
    print("wrote", ROOT / "page.html", f"{len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    build()
