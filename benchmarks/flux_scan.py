"""流量口径诊断（后处理，零成本，不改物理）：沿 x 扫描各列的质量流量 Q(x)=Σ_流体行 ρ·u_x。

用途：每档 finalize 后跑一次，给出**内部列口径**的进出口流量差（判据 <1%）。
背景：入口列 x=0 与出口列 x=lx-1 每步被 BC 覆写，其"流量"分别是规定值与受 BC 影响值，
实测把 0.42% 的真实差放大成 5.4%（2026-10-19 复核方独立复算逐位吻合）。

用法：python benchmarks/flux_scan.py --config D30 --d-lattice 30 --u-mean-l 0.1
      python benchmarks/flux_scan.py --config D30_copy --d-lattice 30 --u-mean-l 0.1 --legacy-outlet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_cylinder import build_case  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="D20")
    p.add_argument("--d-lattice", type=int, default=20)
    p.add_argument("--u-mean-l", type=float, default=0.1)
    p.add_argument("--legacy-outlet", action="store_true")
    a = p.parse_args()

    ckpt = ROOT / "benchmarks" / "grid" / a.config / "ckpt.npz"
    if not ckpt.exists():
        print(f"缺 checkpoint: {ckpt}")
        return 2
    mode = "copy" if a.legacy_outlet else "convective"
    lbm, info, cyl = build_case(a.d_lattice, 100.0, a.u_mean_l, mode)
    with np.load(ckpt, allow_pickle=False) as d:
        lbm.f = d["f"]
        step = int(d["step"])
    rho, ux, _ = lbm.macros()
    lx = info["lx"]

    def q(x: int) -> float:
        return float((rho[x, :] * ux[x, :])[lbm.fluid[x, :]].sum())

    q_in = q(2 if lx > 4 else 0)          # 内部列（入口侧）
    print(f"[{a.config}] 步数 {step}，网格 {info['lx']}x{info['ly']}，出口={mode}")
    print(f"  Q_in（x=2，内部列）= {q_in:.6f}")
    for dx in (1, 2, 3, 5, 10, 20, 50, 100, 200):
        x = lx - dx
        if x <= 2:
            continue
        print(f"  x=lx-{dx:3d}（第 {x} 列）：Q={q(x):.6f}  与 Q_in 相对差 {abs(q(x)-q_in)/q_in*100:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
