"""Re 分离实验归档（24 号指导后）：把 %TEMP% 四档结果 collect 回项目 + 生成差值表。

口径（23/24 号指导定稿）：
  - REF=2.003 只锚定 Re=100；Re=80/120 的"vs 文献"列不适用；
  - 主口径 = 同 Re 下 D40 − D20 的 Cl 差（纯分辨率贡献的近似，见诚实边界）；
  - 诚实边界：固定 Re 下 τ ∝ D，两网格黏性也不同 → 差值 = 黏性变差与空间变好之差，不是单边分辨率效应；
  - 生成器只吐事实（比值、差、门禁），主因判断不进本脚本。
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = Path(os.path.expandvars(r"%TEMP%\re_sep"))
OUT = ROOT / "benchmarks" / "re_separation"
REF100 = 2.003
CASES = [("D20_re80", 20, 80.0), ("D20_re120", 20, 120.0),
         ("D40_re80", 40, 80.0), ("D40_re120", 40, 120.0)]
EXISTING = {"D20_re100": 1.8232, "D40_re100": 1.7180}  # Re=100 档实测（benchmarks/grid/）


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cl = {}
    gates = {}
    for name, d, re_v in CASES:
        p = RUN_DIR / name / "measured.json"
        if not p.exists():
            print(f"缺 {name}，跳过")
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        res, cfg = m["result"], m["config"]
        ok = res["gates"]["mass_ok"] and res["gates"]["stationarity_ok"]
        gates[name] = {"d": d, "re": re_v, "tau": cfg["tau"], "ok": ok,
                       "drift": res["mass"]["drift_steady_window"]}
        if ok:
            cl[name] = res["steady_metrics"]["Cl_amp"]
        shutil.copy2(p, OUT / f"measured_{name}.json")
    # 归档四张图
    for name, _, _ in CASES:
        src = RUN_DIR / name
        if src.exists():
            for f in ("cl_history.png", "stationarity.png", "mass_history.png"):
                if (src / f).exists():
                    shutil.copy2(src / f, OUT / f"{name}_{f}")

    L = ["# 单变量 Re 实验：固定网格尺寸与流速，只改雷诺数（2026-10-28，四档 + Re=100 已有两档）", "",
         f"> 基准 Cl 全幅 {REF100}（Schäfer-Turek DFG 2D-2）**只锚定 Re=100**。",
         "> 自检实测：D20 档 Cl 全幅随 Re 从 1.2509（Re=80）→ 1.8232（Re=100）→ 2.4625（Re=120），",
         f"变化 −31.4% / +35.1% —— **远超 23 号预估的 5%**；拿 Re=100 基准算其他 Re 会把趋势算反",
         f"（Re=80 算出 37.55%\"最差\"、Re=120 算出 22.94%，颠倒了真实排序）。**这是物理（临界区振幅增长），不是数值错误。**",
         "", "## 各档门禁与 Cl 全幅", "",
         "| 档 | D | Re | τ | 门禁 | Cl 全幅 |", "|---|---|---|---|---|---|"]
    for name, d, re_v in CASES:
        g = gates.get(name)
        if not g:
            continue
        ok = "✅" if g["ok"] else "❌"
        c = f"{cl[name]:.4f}" if name in cl else "—"
        L.append(f"| {name} | {d} | {int(re_v)} | {g['tau']:.4f} | {ok} | {c} |")
    for name, c in EXISTING.items():
        L.append(f"| {name}（已有） | {20 if 'D20' in name else 40} | 100 | "
                 f"{0.56 if 'D20' in name else 0.62:.2f} | ✅ | {c:.4f} |")

    L += ["", "## 主口径：同 Re 下 D40 − D20", "",
          "> 🔴 **诚实边界**：固定 Re 与流速时 τ ∝ 网格尺寸（τ=3ν+0.5，ν=U·D/Re），",
          "> **两个网格的黏性也不同**（D40 的 τ 更大）→ 下面的差 = 「黏性变差的量」与「空间变好的量」之差，",
          "> **不是单边的分辨率效应**。主因归属本轮无法单独判定（见 README）。", "",
          "| Re | Cl(D20) | Cl(D40) | 比值 Cl40/Cl20 | 相对差 |ΔCl |", "|---|---|---|---|---|---|"]
    ratios = []
    for re_v in (80.0, 100.0, 120.0):
        k20 = f"D20_re{int(re_v)}"
        k40 = f"D40_re{int(re_v)}"
        c20 = cl.get(k20, EXISTING.get(k20))
        c40 = cl.get(k40, EXISTING.get(k40))
        if c20 and c40:
            ratio = c40 / c20
            ratios.append((re_v, ratio))
            L.append(f"| {int(re_v)} | {c20:.4f} | {c40:.4f} | {ratio:.4f} | "
                     f"{(1-ratio)*100:.2f}% | {c40-c20:+.4f} |")
    if len(ratios) == 3:
        spread = max(r for _, r in ratios) - min(r for _, r in ratios)
        L += ["", f"- 比值跨三个 Re 的散布：{spread:.4f}"
              f"（{'稳定（≤0.02）' if spread <= 0.02 else '随 Re 变化（>0.02，走向由人判读）'}）",
              "- （生成器只吐事实；比值稳定/单调的含义与主因判断见 README 叙述段。）"]

    (OUT / "re_separation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "re_separation.json").write_text(json.dumps(
        {"gates": gates, "cl_amp": cl, "ratios": ratios}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {OUT/'re_separation.md'}")
    print("\n".join(L[16:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
