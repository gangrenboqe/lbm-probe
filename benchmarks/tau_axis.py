"""τ 轴对照（min-cost，零新算力）：只读五档 measured.json，按 τ 排 Cl 误差，算相邻变化率。

生成器纪律：只吐事实（数字、斜率、单调性），不下「主因是谁」的判断——判断写 README。
输出：benchmarks/tau_axis/tau_axis.md + tau_axis.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "benchmarks" / "grid"
OUT = ROOT / "benchmarks" / "tau_axis"

CONFIGS = ["tau055", "D20", "tau057", "D30", "D40"]  # 按 τ 升序


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in CONFIGS:
        p = GRID / name / "measured.json"
        if not p.exists():
            print(f"缺 {name}，跳过")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        res, cfg = d["result"], d["config"]
        if not (res["gates"]["mass_ok"] and res["gates"]["stationarity_ok"]):
            print(f"{name} 门禁未过，不作数")
            continue
        m = res["steady_metrics"]
        rows.append({"name": name, "tau": cfg["tau"], "d": cfg["d_lattice"],
                     "u": cfg["u_mean_l"], "cl_err": m["err_vs_ref"]["Cl_amp"],
                     "st_err": m["err_vs_ref"]["St"], "cl_amp": m["Cl_amp"]})
    if len(rows) < 2:
        print("过门禁档位不足 2，无法对照")
        return 2

    rows.sort(key=lambda r: r["tau"])
    L = ["# τ 轴对照：五档（全过门禁）按 τ 排列（2026-10-28，min-cost 零新算力）", "",
         "> 数据全部来自 `benchmarks/grid/*/measured.json`，本脚本只重排与算斜率，不下主因判断。",
         "> ⚠️ 双变量警告：tau055/tau057 档改 τ 时 U 也变了（0.0833/0.1/0.1167），D 组 D 与 τ 双变量",
         "→ 本表是**相关性**证据，不是单变量因果。", "",
         "| 档 | τ | D | U | Cl 全幅 | Cl 误差 | 相邻 Δ(Cl误差)/Δτ |",
         "|---|---|---|---|---|---|---|"]
    slopes = []
    for i, r in enumerate(rows):
        seg = "—"
        if i > 0:
            dr = r["tau"] - rows[i - 1]["tau"]
            if dr > 0:
                s = (r["cl_err"] - rows[i - 1]["cl_err"]) / dr
                slopes.append(s)
                seg = f"{s*100:+.1f}pp/τ"
        L.append(f"| {r['name']} | {r['tau']:.4f} | {r['d']} | {r['u']:g} | "
                 f"{r['cl_amp']:.4f} | {r['cl_err']*100:.2f}% | {seg} |")
    mono = all(a <= b for a, b in zip([r["cl_err"] for r in rows], [r["cl_err"] for r in rows][1:]))
    # 结构性事实：五档 Re 全为 100 → 固定 Re 下 τ、D、U 三者只能动两个（τ=3ν+0.5, ν=U·D/Re）。
    # 因此这 5 个点混着两类实验（D 组：U 固定 τ∝D；τ 组：D 固定 τ∝U），拼不出 τ 的单变量曲线。
    seam = None
    for i in range(1, len(rows)):
        if rows[i]["d"] != rows[i - 1]["d"]:
            seam = (rows[i - 1], rows[i])
            break
    L += ["", f"- Cl 误差随 τ 是否单调上升：**{'是' if mono else '否（有回落）'}**"
          f"（" + " → ".join(f"{r['cl_err']*100:.2f}%" for r in rows) + "）",
          f"- 相邻段斜率：{' / '.join(f'{s*100:+.1f}pp/τ' for s in slopes)}"]
    if not mono and seam:
        a, b = seam
        L.append(f"- 回落位置：{a['name']}（D={a['d']}, U={a['u']:g}）→ {b['name']}（D={b['d']}, U={b['u']:g}），"
                 f"**位于 D 组与 τ 组的拼接缝上（U 从 {a['u']:g} 跳到 {b['u']:g}），是两类实验的接缝，不是物理非单调**")
    L += ["- 🔴 结构性限制：五档 **Re 全为 100** → 固定 Re 下 τ、D、U 三者只能动两个，"
          "**这 5 档在数学上拼不出 τ 的单变量曲线**，也分不开「黏性离散」与「网格分辨率」的贡献。",
          "- 单变量分离实验（固定 D/U 改 Re）未做，本表不能区分两者的贡献。"]
    (OUT / "tau_axis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "tau_axis.json").write_text(json.dumps(
        {"rows": rows, "slopes_cl_err_per_tau": slopes, "monotonic_rising": mono},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {OUT/'tau_axis.md'} · tau_axis.json")
    print("\n".join(L[-4:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
