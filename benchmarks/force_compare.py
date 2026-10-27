"""控制体积分测力 vs 动量交换法：同一条稳态轨迹上的并行对比（2026-10-27）。

目的：回答「Cl 全幅偏低 5.95%~14% 是否来自动量交换法 + bounce-half 对切向力的系统性低估」。
做法：从 D20 已收敛的 checkpoint（30 物理秒）继续推进 10 秒，每步同时记录两种测力法：
  - 动量交换法（现有 `lbm.step()` 返回值）
  - 控制体 + 动量守恒积分（`lbm.force_cv()`）
两者用同一归一化 2/(U²D)。比较 Cd 均值、Cd 最大、Cl 全幅。

诚实口径：两种方法都可能各有偏差；**结论看两者之差**，不是看谁更接近文献。
如果两者接近 → 「动量交换法低估切向力」这个假设被证伪；如果控制体明显更高 → 支持假设。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lbm_d2q9 import LBM  # noqa: E402
from run_cylinder import build_case, config_id, load_ckpt  # noqa: E402

REF = {"Cd_mean": 3.1811, "Cd_max": 3.2124, "Cl_amp": 2.003}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d-lattice", type=int, default=20)
    p.add_argument("--u-mean-l", type=float, default=0.1)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "force_compare"))
    a = p.parse_args()

    lbm, info, cyl = build_case(a.d_lattice, 100.0, a.u_mean_l, "convective")
    cid = config_id(info, 100.0, "convective", a.sample_every)
    ckpt_path = Path(a.ckpt) if a.ckpt else ROOT / "benchmarks" / "grid" / "D20" / "ckpt.npz"
    state = load_ckpt(ckpt_path, cid)
    if state is None:
        print(f"checkpoint 不可用（缺失或配置指纹不匹配）：{ckpt_path}；cid={cid}")
        return 2
    lbm.f = state["f"]
    step0 = state["step"]
    sps = info["steps_per_second"]
    n_steps = int(round(a.seconds * sps))
    norm = 2.0 / (a.u_mean_l ** 2 * a.d_lattice)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    prev_mom = None
    step = step0
    print(f"从 checkpoint 第 {step} 步（{step/sps:.2f} 物理秒）继续 {a.seconds:g} 秒"
          f"（{n_steps} 步）；网格 {info['lx']}x{info['ly']}，cid={cid}", flush=True)
    for i in range(n_steps):
        fx_me, fy_me = lbm.step()
        step += 1
        fx_cv, fy_cv, prev_mom = lbm.force_cv(prev_mom)
        if step % a.sample_every == 0:
            rows.append({"t": step / sps,
                         "Cd_me": fx_me * norm, "Cl_me": fy_me * norm,
                         "Cd_cv": fx_cv * norm, "Cl_cv": fy_cv * norm})
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n_steps} 步  t={step/sps:.2f}s  "
                  f"Cd_me={rows[-1]['Cd_me']:.4f} Cd_cv={rows[-1]['Cd_cv']:.4f} "
                  f"Cl_me={rows[-1]['Cl_me']:.4f} Cl_cv={rows[-1]['Cl_cv']:.4f}", flush=True)

    t = np.array([r["t"] for r in rows])
    d_me = np.array([r["Cd_me"] for r in rows]); l_me = np.array([r["Cl_me"] for r in rows])
    d_cv = np.array([r["Cd_cv"] for r in rows]); l_cv = np.array([r["Cl_cv"] for r in rows])

    def stats(d, l):
        return {"Cd_mean": float(d.mean()), "Cd_min": float(d.min()), "Cd_max": float(d.max()),
                "Cl_mean": float(l.mean()), "Cl_amp": float(l.max() - l.min())}

    s_me, s_cv = stats(d_me, l_me), stats(d_cv, l_cv)
    res = {"config": {"d": a.d_lattice, "u": a.u_mean_l, "seconds": a.seconds,
                      "ckpt_step": step0, "n": len(rows)},
           "momentum_exchange": s_me, "control_volume": s_cv,
           "reference": REF,
           "diff": {"Cd_mean_abs": abs(s_cv["Cd_mean"] - s_me["Cd_mean"]),
                    "Cl_amp_abs": abs(s_cv["Cl_amp"] - s_me["Cl_amp"]),
                    "Cl_amp_rel": abs(s_cv["Cl_amp"] - s_me["Cl_amp"]) / s_me["Cl_amp"]}}
    (out / "force_compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV
    with open(out / "force_series.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["t", "Cd_me", "Cl_me", "Cd_cv", "Cl_cv"])
        w.writeheader(); w.writerows(rows)

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    ax1.plot(t, d_me, lw=0.9, label=f"动量交换法（均值 {s_me['Cd_mean']:.4f}）")
    ax1.plot(t, d_cv, lw=0.9, label=f"控制体积分（均值 {s_cv['Cd_mean']:.4f}）")
    ax1.axhline(REF["Cd_mean"], color="black", ls="--", lw=1, label=f"文献 {REF['Cd_mean']}")
    ax1.set_ylabel("Cd"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax1.set_title("Cd：两种测力法对比")
    ax2.plot(t, l_me, lw=0.9, label=f"动量交换法（全幅 {s_me['Cl_amp']:.4f}）")
    ax2.plot(t, l_cv, lw=0.9, label=f"控制体积分（全幅 {s_cv['Cl_amp']:.4f}）")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xlabel("物理时间（秒）"); ax2.set_ylabel("Cl"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax2.set_title("Cl：两种测力法对比")
    fig.tight_layout(); fig.savefig(out / "force_compare.png", dpi=150)

    md = f"""# 控制体 vs 动量交换：同一条稳态轨迹上的对比（2026-10-27）

> 配置：D={a.d_lattice}，U={a.u_mean_l}，Re=100，对流出口；从 checkpoint 第 {step0} 步（{step0/sps:.2f} 物理秒）继续 {a.seconds:g} 秒。
> 两种方法用同一归一化 2/(U²D)。**结论看两者之差，不看谁更接近文献。**

| 量 | 动量交换法 | 控制体积分 | 两者绝对差 | 文献 |
|---|---|---|---|---|
| Cd 均值 | {s_me['Cd_mean']:.4f} | {s_cv['Cd_mean']:.4f} | {abs(s_cv['Cd_mean']-s_me['Cd_mean']):.4f} | {REF['Cd_mean']} |
| Cd 最大 | {s_me['Cd_max']:.4f} | {s_cv['Cd_max']:.4f} | {abs(s_cv['Cd_max']-s_me['Cd_max']):.4f} | {REF['Cd_max']} |
| Cl 全幅 | {s_me['Cl_amp']:.4f} | {s_cv['Cl_amp']:.4f} | {abs(s_cv['Cl_amp']-s_me['Cl_amp']):.4f}（{abs(s_cv['Cl_amp']-s_me['Cl_amp'])/s_me['Cl_amp']*100:.2f}%） | {REF['Cl_amp']} |
| Cl 均值 | {s_me['Cl_mean']:.4f} | {s_cv['Cl_mean']:.4f} | {abs(s_cv['Cl_mean']-s_me['Cl_mean']):.4f} | — |

## 判定（按施工单：不管结论是什么都照实写）

"""
    if res["diff"]["Cl_amp_rel"] < 0.01:
        md += ("🔴 **假设被证伪**：两种测力法的 Cl 全幅差 "
               f"{res['diff']['Cl_amp_rel']*100:.2f}%（<1%），说明「动量交换法对切向力系统性低估」"
               "**不是** Cl 偏低的主因。Cl 偏低更可能来自其他环节（边界处理 / 离散误差 / 基准口径），"
               "需另设实验定位。\n")
    elif s_cv["Cl_amp"] > s_me["Cl_amp"]:
        md += (f"🟡 **假设得到支持**：控制体积分的 Cl 全幅比动量交换法高 "
               f"{res['diff']['Cl_amp_rel']*100:.2f}%，方向与「动量交换法低估切向力」一致；"
               f"但控制体值（{s_cv['Cl_amp']:.4f}）距文献（{REF['Cl_amp']}）仍有 "
               f"{abs(s_cv['Cl_amp']-REF['Cl_amp'])/REF['Cl_amp']*100:.2f}% 差距，说明还有别的偏差来源。\n")
    else:
        md += (f"🟡 **假设未获支持**：控制体积分的 Cl 全幅反而比动量交换法低 "
               f"{res['diff']['Cl_amp_rel']*100:.2f}%，方向与假设相反；Cl 偏低的主因需另找。\n")
    md += ("\n## 边界与未验证\n"
           "- 控制体取圆柱外围矩形（margin = 0.5D，不贴壁、不碰出入口），四边做 Π·n 积分 + CV 内动量变化率；\n"
           "- 两种方法并行同一轨迹、同一时间窗，排除「不同轨迹/不同窗口」的干扰；\n"
           "- 只有 10 秒（≈3 个涡脱落周期）才作数；更短的 sanity 窗口不足 1 个周期，不用于结论；\n"
           "- **控制体实现没有用解析解/已发表算例独立标定**，它不能当裁判：\n"
           f"  它的 Cd 均值比文献低 {abs(s_cv['Cd_mean']-REF['Cd_mean'])/REF['Cd_mean']*100:.2f}%，"
           f"而动量交换法只低 {abs(s_me['Cd_mean']-REF['Cd_mean'])/REF['Cd_mean']*100:.2f}%；"
           "Cd 由压差主导，若两法只差切向力，Cd 不该差这么多。\n"
           "- 这里只回答「两法差多少」，不回答「哪个是绝对真值」。\n"
           "- 复算：`python benchmarks/force_compare.py --seconds 10`（从 `benchmarks/grid/D20/ckpt.npz` 续跑）。\n")
    (out / "force_compare.md").write_text(md, encoding="utf-8")

    print(f"\n产物：{out/'force_compare.md'} · force_compare.json · force_series.csv · force_compare.png", flush=True)
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
