"""圆柱绕流算例（DFG 基准 2D-2 / Schäfer–Turek，Re=100）。

几何与参数取自官方基准（物理量）：通道 [0,2.2]×[0,0.41]，圆柱圆心 (0.2,0.2)、半径 0.05（D=0.1），
抛物线入口（最大 1.5 → 平均 U_mean=1.0），ν=0.001，Re = U_mean·D/ν = 100。
参考：Turek & Schäfer, NNFM 52, 547–566 (1996)；数值读自官方页 DFG 2D-2。

🔴 三道门禁（顺序不可跳，任一不过则后面的数字一律不作结论）：
  ① 质量守恒：稳态窗口内流体总质量相对漂移 < 0.1%
  ② 稳态判据：把**最后 1/4 演化时长**切 4 个子窗，各算 Cd 均值，**相邻子窗相对变化 < 0.5%**；
     辅助：Cl 幅值在 4 个子窗之间锁定在 ±3% 内。⚠️ 判据**不看"Cd 是否恒定"**——
     圆柱绕流 Cd 本身带约 2% 的周期振荡，那是物理的一部分。窗口由代码自动取，**不许手动挑**。
  ③ 通过 ①② 后，才对**稳态窗口**做 St(FFT) / Cd_mean / Cl_amp 与文献对照。

可续跑：每 `--ckpt-every` 步存一次全流场 checkpoint（含历史序列），`--run-seconds` 分段推进；
中断后重跑同一命令即自动续跑（配置变了则旧 checkpoint 自动作废）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time as _time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lbm_d2q9 import LBM  # noqa: E402

# 官方基准参考值（DFG 2D-2 / Schäfer–Turek 1996）
REF = {
    "Cd_mean_lv5": 3.1811, "Cd_mean_lv6": 3.1884,
    "Cd_max_lv5": 3.2124, "Cd_max_lv6": 3.2200,
    "Cl_amp": 2.003,                      # = max 0.9859 − min −1.0169（全幅值）
    "St": 0.30303,
    "source": "Turek & Schäfer (1996), NNFM 52, 547-566；数值取自官方基准页 DFG 2D-2",
    "note": "复核方给的 Cd 3.22–3.24 对应 level 6/官方新参考；Cl 0.99–1.01 指半幅值",
}


def build_case(d_lattice: int, re: float, u_mean_l: float, outlet_mode: str = "convective"):
    scale = d_lattice / 0.1
    lx = int(round(2.2 * scale)); ly = int(round(0.41 * scale))
    cx = 0.2 * scale; cy = 0.2 * scale; r = 0.05 * scale
    x = np.arange(lx)[:, None]; y = np.arange(ly)[None, :]
    cyl = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    solid = np.zeros((lx, ly), dtype=bool)
    solid[:, 0] = True; solid[:, -1] = True
    solid |= cyl
    nu_l = u_mean_l * d_lattice / re
    tau = 3.0 * nu_l + 0.5
    yf = np.arange(ly) / (ly - 1)
    u_in = 4.0 * (1.5 * u_mean_l) * yf * (1.0 - yf)
    lbm = LBM(lx, ly, tau, solid, u_in, obstacle=cyl, outlet_mode=outlet_mode)
    return lbm, {"scale": scale, "lx": lx, "ly": ly, "d_lattice": d_lattice,
                 "nu_l": nu_l, "tau": tau, "u_mean_l": u_mean_l,
                 "steps_per_second": d_lattice / u_mean_l * 1.0 / 0.1}, cyl


def config_id(info: dict, re: float, outlet_mode: str, sample_every: int) -> str:
    payload = json.dumps({"d": info["d_lattice"], "re": re, "tau": round(info["tau"], 10),
                          "out": outlet_mode, "se": sample_every}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------- checkpoint（LBM 专用，原子写入 + 配置指纹） ----------
def save_ckpt(path: Path, lbm: LBM, step: int, cid: str, m0: float, h: dict) -> None:
    payload = dict(f=lbm.f, step=step, cid=cid, m0=m0,
                   t=np.array(h["t"]), cd=np.array(h["cd"]), cl=np.array(h["cl"]),
                   t_mass=np.array(h["t_mass"]), mass=np.array(h["mass"]),
                   t_flux=np.array(h["t_flux"]), qin=np.array(h["qin"]),
                   qout=np.array(h["qout"]))
    tmp = path.with_suffix(".npz.tmp")
    try:
        with open(tmp, "wb") as fh:          # 原子写：tmp → rename
            np.savez_compressed(fh, **payload)
        tmp.replace(path)
    except PermissionError:
        # 🔴 实测：沙箱会拒绝在个别目录（如 D40）创建 .tmp，此时退化为直接写目标文件。
        # 丧失原子性，但避免 checkpoint 完全丢失。
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **payload)


def load_ckpt(path: Path, cid: str):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        if str(d["cid"].item()) != cid:
            return None                       # 配置变了 → 旧 checkpoint 作废
        return {"f": d["f"], "step": int(d["step"]), "m0": float(d["m0"]),
                "t": list(d["t"]), "cd": list(d["cd"]), "cl": list(d["cl"]),
                "t_mass": list(d["t_mass"]), "mass": list(d["mass"]),
                "t_flux": list(d["t_flux"]), "qin": list(d["qin"]), "qout": list(d["qout"])}


# ---------- ② 稳态判据（纯后处理；窗口自动取最后 1/4，不许手挑） ----------
def stationarity_report(t: np.ndarray, cd: np.ndarray, cl: np.ndarray,
                        total_steps: int, n_sub: int = 4) -> dict:
    lo = total_steps * 0.75
    hi = float(total_steps)
    edges = np.linspace(lo, hi, n_sub + 1)
    subs = []
    for i in range(n_sub):
        m = (t >= edges[i]) & (t < edges[i + 1])
        subs.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                     "n": int(m.sum()),
                     "Cd_mean": float(cd[m].mean()) if m.any() else None,
                     "Cl_amp": float(cl[m].max() - cl[m].min()) if m.any() else None})
    diffs = []
    for i in range(n_sub - 1):
        a, b = subs[i]["Cd_mean"], subs[i + 1]["Cd_mean"]
        diffs.append(abs(b - a) / abs(a) if a else None)
    amps = [s["Cl_amp"] for s in subs if s["Cl_amp"] is not None]
    amp_spread = (max(amps) - min(amps)) / (sum(amps) / len(amps)) if amps else None
    cd_ok = bool(diffs) and all(d is not None and d < 5e-3 for d in diffs)
    cl_ok = amp_spread is not None and amp_spread < 0.03
    return {"window": [lo, hi], "subwindows": subs,
            "adjacent_diff": diffs, "cd_locked": cd_ok,
            "cl_amp_spread": amp_spread, "cl_locked": cl_ok,
            "passed": bool(cd_ok and cl_ok)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="D2Q9 LBM 圆柱绕流（DFG 基准 Re=100；含三道门禁 + 可续跑）")
    p.add_argument("--d-lattice", type=int, default=20)
    p.add_argument("--re", type=float, default=100.0)
    p.add_argument("--u-mean-l", type=float, default=0.1)
    p.add_argument("--total-seconds", type=float, default=30.0, help="总目标物理秒数")
    p.add_argument("--run-seconds", type=float, default=10.0, help="本次调用推进多少物理秒（可分段续跑）")
    p.add_argument("--finalize", action="store_true", help="只做门禁分析 + 出图，不推进")
    p.add_argument("--ckpt-every", type=int, default=10000, help="每多少步存一次全流场 checkpoint")
    p.add_argument("--progress-every", type=int, default=2000)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--mass-every", type=int, default=10)
    p.add_argument("--flux-every", type=int, default=100,
                   help="记录进出口流量（时均用）的间隔步数")
    p.add_argument("--n-sub", type=int, default=4, help="稳态判据的子窗个数")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--legacy-outlet", action="store_true", help="旧整列拷贝出口（仅归因对照实验）")
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    outlet_mode = "copy" if args.legacy_outlet else "convective"
    lbm, info, cyl = build_case(args.d_lattice, args.re, args.u_mean_l, outlet_mode)
    sps = info["steps_per_second"]
    total_steps = int(round(args.total_seconds * sps))
    cid = config_id(info, args.re, outlet_mode, args.sample_every)
    norm = 2.0 / (info["u_mean_l"] ** 2 * info["d_lattice"])
    ckpt_path = out_dir / "ckpt.npz"

    state = load_ckpt(ckpt_path, cid)
    if state:
        lbm.f = state["f"]; step = state["step"]; m0 = state["m0"]
        h = {"t": state["t"], "cd": state["cd"], "cl": state["cl"],
             "t_mass": state["t_mass"], "mass": state["mass"],
             "t_flux": state["t_flux"], "qin": state["qin"], "qout": state["qout"]}
        print(f"续跑：从 checkpoint 第 {step} 步（{step/sps:.2f} 物理秒）继续；"
              f"已有 Cd/Cl 采样 {len(h['cd'])} 点", flush=True)
    else:
        step = 0; m0 = lbm.fluid_mass()
        h = {"t": [], "cd": [], "cl": [], "t_mass": [], "mass": [],
             "t_flux": [], "qin": [], "qout": []}
        print("无可用 checkpoint → 从 0 开始", flush=True)
    print(f"算例：D_l={info['d_lattice']} → 网格 {info['lx']}×{info['ly']}，ν_l={info['nu_l']:.5f}，"
          f"τ={info['tau']:.4f}（Re={args.re:g} 反推），出口={outlet_mode}，"
          f"目标 {total_steps} 步（{args.total_seconds:g} 秒）；1 秒={sps:.0f} 步；配置指纹 {cid}", flush=True)

    if not args.finalize:
        target = min(step + int(round(args.run_seconds * sps)), total_steps)
        t0 = _time.time(); diverged = False
        while step < target:
            fx, fy = lbm.step(); step += 1
            m = lbm.fluid_mass()
            if step % args.mass_every == 0:
                h["mass"].append(m); h["t_mass"].append(step)
            if (not np.isfinite(m)) or m > 10.0 * m0 or m < 0.1 * m0:
                diverged = True
                print(f"  🔴 发散早停：第 {step} 步 M={m!r}（初值 {m0:.6e}）", flush=True)
                break
            if step % args.sample_every == 0:
                h["cd"].append(fx * norm); h["cl"].append(fy * norm); h["t"].append(step)
            if step % args.flux_every == 0:            # 时均流量（瞬时值受涡脱落周期影响，不可用）
                h["qin"].append(lbm.inlet_flux()); h["qout"].append(lbm.outlet_flux())
                h["t_flux"].append(step)
            if args.progress_every and step % args.progress_every == 0:
                el = _time.time() - t0
                print(f"  {step}/{total_steps} 步（{100*step/total_steps:4.1f}%）"
                      f" t={step/sps:5.2f}s  M={m:.6e}（{100*(m-m0)/m0:+.3f}%）"
                      f" Cd={h['cd'][-1]:+.3f} Cl={h['cl'][-1]:+.3f}"
                      f" 本段用时 {el:5.1f}s", flush=True)
            if step % args.ckpt_every == 0:
                save_ckpt(ckpt_path, lbm, step, cid, m0, h)
                print(f"  💾 checkpoint 已存（第 {step} 步）", flush=True)
        save_ckpt(ckpt_path, lbm, step, cid, m0, h)
        print(f"本段结束：第 {step}/{total_steps} 步（{step/sps:.2f}/{args.total_seconds:g} 物理秒）"
              f"；已存 checkpoint（{ckpt_path}）；发散={diverged}", flush=True)
        if step < total_steps:
            print(f"⏭ 还需 {total_steps-step} 步 ≈ {(total_steps-step)/sps:.1f} 秒 → 再跑一次同命令即可续跑", flush=True)
            return 0

    # ---------- 门禁分析（自动窗口，不手挑） ----------
    # 🔴 窗口分母以**实际推进步数**为准（否则 --finalize 用 CLI 目标值会让窗口错位；
    #    自检时已实测到该 bug：跑 400 步却按 60000 步算窗口 → 稳态窗口内 0 个采样点）
    total_steps = step
    t = np.array(h["t"], dtype=float); cd = np.array(h["cd"]); cl = np.array(h["cl"])
    tm = np.array(h["t_mass"], dtype=float); ma = np.array(h["mass"])
    print(f"\n=== 三道门禁（顺序执行，任一不过则后面不作结论）===", flush=True)

    # ① 质量守恒：稳态窗口（最后 1/4 演化时长）内的相对漂移
    lo = total_steps * 0.75
    win = tm >= lo
    m_ref = float(ma[win].mean()) if win.any() else float(ma.mean())
    drift_steady = float(np.max(np.abs(ma[win] - m_ref)) / m_ref) if win.any() else float("nan")
    m_ref_all = float(ma[max(int(len(ma)*0.1), 1):].mean())
    drift_all = float(np.max(np.abs(ma[max(int(len(ma)*0.1), 1):] - m_ref_all)) / m_ref_all)
    # 🔴 流量指标：用**稳态窗口内的时均**（瞬时值受涡脱落周期影响，上一轮报的 5.714% 就是瞬时取样）
    tf = np.array(h["t_flux"], dtype=float); qa_in = np.array(h["qin"]); qa_out = np.array(h["qout"])
    fw = tf >= lo
    flux_in = float(qa_in[fw].mean()) if fw.any() else float(qa_in.mean())
    flux_out = float(qa_out[fw].mean()) if fw.any() else float(qa_out.mean())
    flux_err = abs(flux_out - flux_in) / abs(flux_in)
    flux_inst = abs(qa_out[-1] - qa_in[-1]) / abs(qa_in[-1]) if len(qa_in) else float("nan")
    gate1 = drift_steady < 1e-3
    print(f"① 质量守恒：稳态窗口漂移 {drift_steady*100:.4f}%（判据 <0.1% → {'通过' if gate1 else '不通过'}）；"
          f"全程漂移 {drift_all*100:.4f}%", flush=True)
    print(f"   流量（稳态窗口时均，n={int(fw.sum())} 个采样点）：Q_in={flux_in:.6f}，Q_out={flux_out:.6f}，"
          f"相对差 {flux_err*100:.3f}%（判据 <1% → {'通过' if flux_err < 0.01 else '不通过'}）；"
          f"对照：末点瞬时值差 {flux_inst*100:.3f}%（旧口径，仅参考）", flush=True)

    # ② 稳态判据
    st_rep = stationarity_report(t, cd, cl, total_steps, args.n_sub)
    print(f"② 稳态判据：窗口 = 最后 1/4（第 {st_rep['window'][0]:.0f}–{st_rep['window'][1]:.0f} 步），逐子窗 Cd 均值：", flush=True)
    for i, s in enumerate(st_rep["subwindows"]):
        print(f"     子窗{i+1} [{s['lo']:.0f},{s['hi']:.0f}): Cd均值={s['Cd_mean']:.4f}  "
              f"Cl幅值={s['Cl_amp']:.4f}  n={s['n']}", flush=True)
    print("     相邻相对变化：" + "，".join(
        f"{d*100:.3f}%" if d is not None else "n/a" for d in st_rep["adjacent_diff"])
        + f"（判据 <0.5% → {'锁定' if st_rep['cd_locked'] else '未收敛'}）", flush=True)
    print(f"     Cl 幅值跨子窗散布 {st_rep['cl_amp_spread']*100:.2f}%（判据 ±3% → "
          f"{'锁定' if st_rep['cl_locked'] else '未收敛'}）", flush=True)
    gate2 = st_rep["passed"]
    print(f"② 结论：{'✅ 通过（稳态锁定）' if gate2 else '❌ 未通过（未收敛）'}", flush=True)

    # ③ 只有 ①② 都过才报数
    result = {"gates": {"mass_ok": gate1, "stationarity_ok": gate2},
              "mass": {"drift_steady_window": drift_steady, "drift_all": drift_all,
                       "flux_rel_err_steady_avg": flux_err, "flux_rel_err_inst_final": flux_inst,
                       "qin_avg": flux_in, "qout_avg": flux_out, "M0": m0},
              "stationarity": st_rep, "reference": REF}
    if gate1 and gate2:
        w = (t >= lo)
        cl_w, cd_w, t_w = cl[w], cd[w], t[w]
        sig = cl_w - cl_w.mean()

        # ── 涡脱落频率：两种方法都算，差异如实报（2026-10-20 修正）──
        # 方法 A：FFT 主频。⚠️ 频率分辨率 = 1/窗口长，稳态窗口仅 7.5 秒时
        #   量化误差可达 ~5%，单独使用会低估频率（曾造成 St 恒定偏低 7.6%）。
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        k = int(np.argmax(spec[1:]) + 1)
        f_fft = (k / len(sig)) / args.sample_every

        # 方法 B：过零计数。用「总跨度 / 半周期数」而非逐周期平均，抗抖动；
        #   不受 FFT 频率分辨率限制，是本次采用的**主口径**。
        zero = np.where(np.diff(np.signbit(sig)))[0]
        f_zero = float("nan")
        n_zero = len(zero)
        if n_zero >= 4:
            span = zero[-1] - zero[0]
            f_zero = (n_zero / (2.0 * span)) / args.sample_every
        f_per_step = f_zero if f_zero == f_zero else f_fft

        # ── 有效直径：bounce-back 壁在固体掩码边缘外半格（2026-10-20 修正）──
        # 掩码取 (x-cx)²+(y-cy)² <= r²，r = 0.05*scale → 覆盖 2r+1 个格点；
        # 壁面在掩码外 0.5 格 → 有效直径 D_eff = D + 1（格）。
        d_eff = info["d_lattice"] + 1
        st_num = f_per_step * d_eff / info["u_mean_l"]
        st_fft = f_fft * d_eff / info["u_mean_l"]
        result["steady_metrics"] = {
            "Cd_mean": float(cd_w.mean()), "Cd_min": float(cd_w.min()), "Cd_max": float(cd_w.max()),
            "Cl_amp": float(cl_w.max() - cl_w.min()), "St": float(st_num),
            "St_fft_only": float(st_fft), "d_eff_used": d_eff,
            "freq_per_step_zero": float(f_per_step), "freq_per_step_fft": float(f_fft),
            "err_vs_ref": {"Cd_mean_lv5": abs(cd_w.mean() - REF["Cd_mean_lv5"]) / REF["Cd_mean_lv5"],
                            "Cd_max_lv5": abs(cd_w.max() - REF["Cd_max_lv5"]) / REF["Cd_max_lv5"],
                            "Cd_max_lv6": abs(cd_w.max() - REF["Cd_max_lv6"]) / REF["Cd_max_lv6"],
                            "Cl_amp": abs((cl_w.max() - cl_w.min()) - REF["Cl_amp"]) / REF["Cl_amp"],
                            "St": abs(st_num - REF["St"]) / REF["St"],
                            "St_fft_only": abs(st_fft - REF["St"]) / REF["St"]}}
        m = result["steady_metrics"]
        print(f"③ 稳态段对照文献：Cd_mean={m['Cd_mean']:.4f}（误差 {m['err_vs_ref']['Cd_mean_lv5']*100:.2f}%）；"
              f"Cd_max={m['Cd_max']:.4f}（误差 lv5 {m['err_vs_ref']['Cd_max_lv5']*100:.2f}% / "
              f"lv6 {m['err_vs_ref']['Cd_max_lv6']*100:.2f}%）；"
              f"Cl_amp={m['Cl_amp']:.4f}（误差 {m['err_vs_ref']['Cl_amp']*100:.2f}%）", flush=True)
        print(f"   St（过零法 + D_eff={d_eff}）= {m['St']:.4f}（误差 {m['err_vs_ref']['St']*100:.2f}%）"
              f"｜对照：St（仅 FFT + D_eff={d_eff}）= {m['St_fft_only']:.4f}"
              f"（误差 {m['err_vs_ref']['St_fft_only']*100:.2f}%）", flush=True)
    else:
        print(f"③ 前两道门禁未全过 → **本轮不报 St/Cd 作为结论**"
              f"（如实记为：未收敛，守恒漂移 {drift_steady*100:.4f}%，"
              f"相邻子窗最大变化 {max([d for d in st_rep['adjacent_diff'] if d is not None], default=float('nan'))*100:.3f}%）", flush=True)

    (out_dir / "measured.json").write_text(
        json.dumps({"config": {**info, "re": args.re, "total_steps": total_steps,
                                "outlet_mode": outlet_mode, "cid": cid},
                    "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "history.npz", t=t, cd=cd, cl=cl, t_mass=tm, mass=ma,
                        config=json.dumps({"info": info, "re": args.re, "cid": cid}, ensure_ascii=False))

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        tsec = t / sps

        fig1, ax1 = plt.subplots(figsize=(8.6, 4.4))
        ax1.plot(tm / sps, ma, lw=1.2, color="#2ca02c")
        ax1.axhline(m_ref, color="black", ls="--", lw=1, label=f"稳态窗口参考 M={m_ref:.5e}")
        ax1.axvline(lo / sps, color="red", ls=":", lw=1.2, label="稳态窗口起点（75%）")
        ax1.set_yscale("log")
        ax1.set_xlabel("物理时间（秒）"); ax1.set_ylabel("流体总质量 M（对数轴）")
        ax1.set_title(f"① 质量守恒自检：稳态窗口漂移 {drift_steady*100:.4f}%（判据 <0.1% → "
                      f"{'通过' if gate1 else '不通过'}）")
        ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=8)
        fig1.tight_layout(); fig1.savefig(out_dir / "mass_history.png", dpi=150)

        fig2, ax2 = plt.subplots(figsize=(8.6, 4.4))
        ax2.plot(tsec, cd, lw=0.9, color="#1f77b4", label="Cd")
        for i, s in enumerate(st_rep["subwindows"]):
            ax2.hlines(s["Cd_mean"], s["lo"] / sps, s["hi"] / sps, color="red", lw=2.2,
                       label="子窗 Cd 均值" if i == 0 else "_")
        ax2.axvline(lo / sps, color="black", ls=":", lw=1.2, label="稳态窗口起点")
        ax2.set_xlabel("物理时间（秒）"); ax2.set_ylabel("Cd")
        ax2.set_title(f"② 稳态判据：最后 1/4 切 {args.n_sub} 个子窗，相邻 Cd 均值变化 "
                      f"{'<0.5% → 锁定' if st_rep['cd_locked'] else '≥0.5% → 未收敛'}")
        ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
        fig2.tight_layout(); fig2.savefig(out_dir / "stationarity.png", dpi=150)

        fig3, ax3 = plt.subplots(figsize=(8.6, 4.2))
        ax3.plot(tsec, cl, lw=0.9, color="#1f77b4")
        ax3.set_xlabel("物理时间（秒）"); ax3.set_ylabel("Cl")
        ax3.set_title("升力系数时程（稳态窗口内应呈稳定周期）")
        ax3.grid(alpha=0.3)
        fig3.tight_layout(); fig3.savefig(out_dir / "cl_history.png", dpi=150)

        vor = lbm.vorticity().T
        vmax = float(np.percentile(np.abs(vor[~cyl.T]), 99.5))
        fig4, ax4 = plt.subplots(figsize=(9.2, 2.6))
        im = ax4.imshow(vor, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        extent=[0, info["lx"], 0, info["ly"]], aspect="equal")
        ax4.contour(np.arange(info["lx"]), np.arange(info["ly"]), cyl.T.astype(float),
                    levels=[0.5], colors="black", linewidths=1.2)
        ax4.set_title(f"涡量场（t≈{step/sps:.1f} 物理秒）")
        ax4.set_xlabel("x（格子）"); ax4.set_ylabel("y（格子）")
        fig4.colorbar(im, ax=ax4, label="涡量")
        fig4.tight_layout(); fig4.savefig(out_dir / "vorticity.png", dpi=150)

    print(f"\n产物：{out_dir/'measured.json'} · history.npz"
          + ("" if args.no_plot else " · mass_history.png · stationarity.png · cl_history.png · vorticity.png"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

