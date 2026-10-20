"""D2Q9 格子玻尔兹曼求解器（BGK 碰撞 + 半步反弹边界）。

标准 D2Q9：二维、9 个离散速度方向。每步两步：
  1) 碰撞（collide）：局部松弛到平衡态，τ 由雷诺数反推；
  2) 迁移（stream）：每个格点把分布函数搬给邻居——**只依赖邻居，天然并行**（这也是它适合 GPU 的原因）。

🔴 2026-10-19 返工修复（依据复核结论，根因：碰撞未跳过固体格子）：
  - **碰撞只在流体格子做**，固体格子分布函数恒置 0（此前对整个网格碰撞，
    固体格子的垃圾分布函数会被逐步放大——实测全网格 Σρ 在 50 步内就跑到 −3.5×10⁵）；
  - **宏观量只在流体格子统计**（ρ 在固体处置 1、u 置 0，避免除零产生 NaN/Inf）；
  - 新增质量守恒自检接口：`fluid_mass()` / `inlet_flux()` / `outlet_flux()`
    —— 从此数值模拟先看守恒曲线，守恒不过不报数。

边界处理：固壁与圆柱用半步反弹；入口按抛物线速度剖面给平衡态；
出口用对流式外推（出流分量零梯度、入流分量用内部速度的平衡态，比"整列拷贝"更合理）。

圆柱受力：动量交换法，只对「流体→圆柱」链接累加 2·c_i·f_i，再无量纲化为 Cd/Cl。
"""
from __future__ import annotations

import numpy as np

C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int64)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CX = C[:, 0]


def _shift(arr: np.ndarray, c: tuple[int, int]) -> np.ndarray:
    """返回 M，使 M[x] = arr[x + c]（越界处取 0）。不环绕，避免左右边界被错误连通。"""
    cx, cy = int(c[0]), int(c[1])
    nx, ny = arr.shape
    M = np.zeros_like(arr)
    sx = slice(max(0, -cx), nx - max(0, cx))
    dx = slice(max(0, cx), nx - max(0, -cx))
    sy = slice(max(0, -cy), ny - max(0, cy))
    dy = slice(max(0, cy), ny - max(0, -cy))
    M[sx, sy] = arr[dx, dy]
    return M


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    f = np.empty((9,) + rho.shape)
    u2 = ux * ux + uy * uy
    for i in range(9):
        cu = C[i, 0] * ux + C[i, 1] * uy
        f[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
    return f


class LBM:
    """D2Q9 BGK 求解器。f 形状 (9, nx, ny)；固体格子恒为 0。"""

    def __init__(self, nx: int, ny: int, tau: float, solid: np.ndarray,
                 u_inlet: np.ndarray, obstacle: np.ndarray | None = None,
                 outlet_mode: str = "convective"):
        self.nx, self.ny = nx, ny
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.solid = solid
        self.fluid = ~solid
        self.obstacle = solid if obstacle is None else obstacle
        self.u_inlet = u_inlet
        self.outlet_mode = outlet_mode       # "convective"（默认）| "copy"（旧实现，仅供对照实验）
        ny_ = ny
        self.f = equilibrium(np.ones((nx, ny_)), np.zeros((nx, ny_)), np.zeros((nx, ny_)))
        self.f[:, self.solid] = 0.0          # 🔴 固体不存分布函数
        self.inlet_rows = self.fluid[0, :]   # 入口列里哪些行是流体
        self.outlet_rows = self.fluid[-1, :]

    # ---------- 宏观量 ----------
    def macros(self, mask_solid_to_quiet: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """流体格子的 ρ/u；固体格处置 1/0 以免除零（不参与物理）。"""
        rho = self.f.sum(axis=0)
        ux = (self.f * C[:, 0, None, None]).sum(axis=0)
        uy = (self.f * C[:, 1, None, None]).sum(axis=0)
        safe = np.where(rho > 0, rho, 1.0)
        ux = ux / safe
        uy = uy / safe
        if mask_solid_to_quiet:
            rho = np.where(self.fluid, rho, 1.0)
            ux = np.where(self.fluid, ux, 0.0)
            uy = np.where(self.fluid, uy, 0.0)
        return rho, ux, uy

    def fluid_mass(self) -> float:
        """🔴 质量守恒自检：**只统计流体格子**的 Σρ（固体为 0，本来就不该计入）。"""
        return float(self.f[:, self.fluid].sum())

    def inlet_flux(self) -> float:
        """🔴 入口质量流量 Σ ρ·u_x。**取第 1 列（内部列）**，不取 x=0：
        x=0 每步被入口 BC 覆写为平衡态（速度=规定剖面），其"流量"是规定值不是实流。"""
        rho, ux, _ = self.macros()
        x = 1 if self.nx > 3 else 0
        return float((rho[x, :] * ux[x, :])[self.fluid[x, :]].sum())

    def outlet_flux(self) -> float:
        """🔴 出口质量流量。**取倒数第 2 列（内部列）**，不取最后一列：
        最后一列每步被出口 BC 覆写，速度不是真实出流（实测会把 0.4% 的真实差放大成 5.4%）。"""
        rho, ux, _ = self.macros()
        x = self.nx - 2 if self.nx > 3 else self.nx - 1
        return float((rho[x, :] * ux[x, :])[self.fluid[x, :]].sum())

    # ---------- 单步 ----------
    def step(self) -> tuple[float, float]:
        rho, ux, uy = self.macros()
        feq = equilibrium(rho, ux, uy)
        f_post = self.f - self.omega * (self.f - feq)      # 碰撞
        f_post[:, self.solid] = 0.0                        # 🔴 固体不参与碰撞

        # 动量交换法：只对「流体→圆柱」链接求和（不含上下壁）
        fx = fy = 0.0
        for i in range(1, 9):
            through = self.fluid & _shift(self.obstacle, (C[i, 0], C[i, 1]))
            if through.any():
                contrib = f_post[i][through].sum()
                fx += 2.0 * C[i, 0] * contrib
                fy += 2.0 * C[i, 1] * contrib

        # 迁移 + 半步反弹（反射只写在流体格）
        f_new = np.zeros_like(self.f)
        for i in range(9):
            f_new[i] = _shift(f_post[i], (-C[i, 0], -C[i, 1]))
        for i in range(1, 9):
            through = self.fluid & _shift(self.solid, (C[i, 0], C[i, 1]))
            f_new[OPP[i]][through] = f_post[i][through]

        # 入口：抛物线剖面平衡态（只写流体行）
        feq_in = equilibrium(np.ones(self.ny), self.u_inlet, np.zeros(self.ny))
        f_new[:, 0, self.inlet_rows] = feq_in[:, self.inlet_rows]

        # 出口：convective（默认）= 出流分量零梯度 + 入流分量用内部速度的平衡态；
        #       copy = 旧实现（整列拷贝），仅用于「归因对照实验」
        if self.outlet_mode == "copy":
            f_new[:, -1, self.outlet_rows] = f_new[:, -2, self.outlet_rows]
        else:
            feq_out = equilibrium(np.ones(self.ny), ux[-2, :], uy[-2, :])
            for i in range(9):
                if CX[i] > 0:
                    f_new[i, -1, self.outlet_rows] = f_new[i, -2, self.outlet_rows]
                elif CX[i] < 0:
                    f_new[i, -1, self.outlet_rows] = feq_out[i][self.outlet_rows]

        f_new[:, self.solid] = 0.0                         # 固体恒为 0
        self.f = f_new
        return fx, fy

    def vorticity(self) -> np.ndarray:
        _, ux, uy = self.macros()
        return np.gradient(uy, axis=0) - np.gradient(ux, axis=1)
