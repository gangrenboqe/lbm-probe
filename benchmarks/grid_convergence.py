"""收官分析：读各配置的 measured.json → 收敛曲线 + 汇总表 + 误差来源判定。

用法：python benchmarks/grid_convergence.py
读取目录：benchmarks/grid/{D20,D30,D40,tau055,tau057,D30_copy}/measured.json
输出：benchmarks/grid/grid_convergence.png · benchmarks/grid/summary.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GRID = ROOT / "grid"
CONFIGS = ["D20", "D30", "D40"]
EXTRA = ["tau055", "tau057", "D30_copy"]


def load(name: str) -> dict | None:
    p = GRID / name / "measured.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    res = d["result"]
    cfg = d["config"]
    row = {"name": name, "d": cfg["d_lattice"], "u": cfg["u_mean_l"], "tau": cfg["tau"],
           "mesh": f"{cfg['lx']}x{cfg['ly']}", "outlet": cfg.get("outlet_mode", "?"),
           "mass_ok": res["gates"]["mass_ok"], "stat_ok": res["gates"]["stationarity_ok"],
           "drift": res["mass"]["drift_steady_window"],
           "flux_avg": res["mass"].get("flux_rel_err_steady_avg"),
           "flux_inst": res["mass"].get("flux_rel_err_inst_final"),
           "sub_cd": [s["Cd_mean"] for s in res["stationarity"]["subwindows"]],
           "adj": res["stationarity"]["adjacent_diff"],
           "cl_spread": res["stationarity"]["cl_amp_spread"]}
    if "steady_metrics" in res:
        m = res["steady_metrics"]
        row.update({"Cd_mean": m["Cd_mean"], "Cd_max": m["Cd_max"], "Cl_amp": m["Cl_amp"],
                    "St": m["St"], "err": m["err_vs_ref"]})
    else:
        row.update({"Cd_mean": None, "Cd_max": None, "Cl_amp": None, "St": None, "err": None})
    return row


def fmt(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def main() -> int:
    rows = [r for r in (load(n) for n in CONFIGS + EXTRA) if r]
    if not rows:
        print("没有找到任何 measured.json"); return 2
    ref = {"St": 0.30303, "Cd_mean": 3.1811, "Cd_max": 3.2124, "Cl_amp": 2.003}

    # -------- 汇总表 --------
    L = ["# 收官汇总：网格收敛 + 排除性实验", "",
         "> 全部配置同口径：`--total-seconds 30 --re 100 --n-sub 4`，三段 × 10 秒推进（checkpoint 可续跑）。",
         "> ⚠️ τ 在 LBM 中由 Re 唯一决定（τ = 3ν + 0.5），**不是自由参数**：",
         "> - 网格收敛档：固定 Re 与 U（Ma 不变）→ **τ 随 D 变化是必然**；",
         "> - τ 敏感性档：**保持 Re=100**、由 τ 反推 U（U = Re·ν/D）——这是唯一能把 τ 与 Re 分离的口径。", "",
         "## 一、逐档结果与门禁", "",
         "| 配置 | D | U(格子) | τ | 网格 | 出口 | ①守恒 | ②稳态 | Cd_mean | Cd_max | Cl全幅 | St |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['d']} | {r['u']:g} | {r['tau']:.4f} | {r['mesh']} | {r['outlet']} | "
                 f"{'✅' if r['mass_ok'] else '❌'} | {'✅' if r['stat_ok'] else '❌'} | "
                 f"{fmt(r['Cd_mean'])} | {fmt(r['Cd_max'])} | {fmt(r['Cl_amp'])} | {fmt(r['St'])} |")
    L += ["", "## 二、相对误差（vs 文献：St 0.30303 / Cd_mean 3.1811 / Cd_max 3.2124 / Cl 全幅 2.003）", "",
          "| 配置 | St 误差 | Cd_mean 误差 | Cd_max 误差 | Cl 全幅误差 |", "|---|---|---|---|---|"]
    for r in rows:
        e = r["err"]
        if e:
            L.append(f"| {r['name']} | {e['St']*100:.2f}% | {e['Cd_mean_lv5']*100:.2f}% | "
                     f"{e['Cd_max_lv5']*100:.2f}% | {e['Cl_amp']*100:.2f}% |")
        else:
            L.append(f"| {r['name']} | 门禁未过，不作数（仅记录） | | | |")
    L += ["", "## 三、门禁明细（子窗 Cd 均值 + 相邻变化%）", "",
          "| 配置 | 四子窗 Cd 均值 | 相邻变化% | Cl 幅值散布% | 守恒漂移% | 流量差%(时均) | 流量差%(末点瞬时) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        sub = " / ".join(f"{v:.4f}" for v in r["sub_cd"]) if r["sub_cd"] else "—"
        adj = " / ".join(f"{d*100:.3f}" for d in r["adj"] if d is not None) if r["adj"] else "—"
        L.append(f"| {r['name']} | {sub} | {adj} | "
                 f"{('%.3f' % (r['cl_spread']*100)) if r['cl_spread'] is not None else '—'} | "
                 f"{r['drift']*100:.4f} | {('%.3f' % (r['flux_avg']*100)) if r['flux_avg'] is not None else '—'} | "
                 f"{('%.3f' % (r['flux_inst']*100)) if r['flux_inst'] is not None else '—'} |")

    # -------- 判定（按施工单给出的三种情况） --------
    g = [r for r in rows if r["name"] in CONFIGS]
    ok = [r for r in g if r["mass_ok"] and r["stat_ok"]]
    L += ["", "## 四、误差来源判定（按施工单的三分支）", ""]
    if len(ok) < 3:
        miss = [r["name"] for r in g if not (r["mass_ok"] and r["stat_ok"])]
        L.append(f"🔴 **无法完成收敛判定**：三档中 {', '.join(miss)} 未过门禁 → 按要求「该档不作数」。"
                 f"下面只用过了门禁的档位做趋势描述，**不下收敛结论**。")
    if len(ok) >= 2:
        st_errs = {r["name"]: r["err"]["St"] for r in ok}
        cd_errs = {r["name"]: r["err"]["Cd_max_lv5"] for r in ok}
        cl_errs = {r["name"]: r["err"]["Cl_amp"] for r in ok}
        L += ["", f"过了门禁的档位误差：St " + "、".join(f"{k} {v*100:.2f}%" for k, v in st_errs.items())
              + "；Cd_max " + "、".join(f"{k} {v*100:.2f}%" for k, v in cd_errs.items())
              + "；Cl 全幅 " + "、".join(f"{k} {v*100:.2f}%" for k, v in cl_errs.items())]
        if len(ok) == 3:
            ds = [r["d"] for r in ok]; order = np.argsort(ds)
            st_seq = [st_errs[f"D{ds[i]}"] for i in order]
            cl_seq = [cl_errs[f"D{ds[i]}"] for i in order]
            def trend(seq):
                return "减小" if all(a >= b for a, b in zip(seq, seq[1:])) and seq[-1] < seq[0] else (
                    "增大" if all(a <= b for a, b in zip(seq, seq[1:])) and seq[-1] > seq[0] else "无规律")
            L += ["", f"- St 误差随 D 的趋势：**{trend(st_seq)}**（{st_seq[0]*100:.2f}% → {st_seq[-1]*100:.2f}%）",
                  f"- Cl 全幅误差随 D 的趋势：**{trend(cl_seq)}**（{cl_seq[0]*100:.2f}% → {cl_seq[-1]*100:.2f}%）"]
    L += ["", "> 判定口径（施工单原文）：误差随 D 明显减小 → ✅ 分辨率问题，GPU 上大网格可解决；"
          "误差基本不变 → 🔴 不是分辨率问题（怀疑出口 BC / 判据）；误差变大/无规律 → 🟡 实现有问题，停 GPU 计划。",
          "> ⛔ 不许因\"趋势大致单调\"就写\"已验证收敛\"——上面给的是逐档具体数字。"]

    (GRID / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # -------- 收敛曲线（横轴 1/D，越细网格越靠右） --------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    specs = [("St", ref["St"], "St"), ("Cd_max", ref["Cd_max"], "Cd_max"),
             ("Cl_amp", ref["Cl_amp"], "Cl 全幅")]
    for ax, (key, refv, title) in zip(axes, specs):
        xs, ys, colors = [], [], []
        for r in g:
            if r[key] is None:
                continue
            xs.append(1.0 / r["d"]); ys.append(r[key])
            colors.append("#2ca02c" if (r["mass_ok"] and r["stat_ok"]) else "#d62728")
        ax.scatter(xs, ys, s=70, c=colors, zorder=3)
        for x, y, r in zip(xs, ys, [r for r in g if r[key] is not None]):
            ax.annotate(f"D{r['d']}", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.axhline(refv, color="black", ls="--", lw=1.2, label=f"文献 {refv:.4g}")
        ax.set_xlabel("1/D（越右网格越细）"); ax.set_ylabel(title)
        ax.set_title(f"{title} 网格收敛（绿=门禁过，红=门禁未过）")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(GRID / "grid_convergence.png", dpi=150)
    print(f"已写：{GRID/'summary.md'} · {GRID/'grid_convergence.png'}")
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
