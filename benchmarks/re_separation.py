"""单变量 τ 分离实验（22 号施工单第四节）：固定 D 与 U，只改 Re（Re → ν → τ）。

四档（Re=100 的 D20/D40 已有，不重跑）：
  D=20, U=0.1, Re=80  → τ=0.575
  D=20, U=0.1, Re=120 → τ=0.55
  D=40, U=0.1, Re=80  → τ=0.65
  D=40, U=0.1, Re=120 → τ=0.60

判据照旧三道门禁（质量守恒 <0.1% → 稳态子窗锁定 → 才准报数），不许因为是"补档"就跳过。
对照口径：只用 Cl 相对偏差（Cd 随 Re 变，不能比绝对值），并算 Cl·Re² 交叉验证。
生成器纪律：只吐事实（数字、门禁、单调性），主因判断写 README。

⚠️ 后台进程对 E: 只读（PermissionError）→ 计算目录默认 %TEMP%\\re_sep\\，
跑完后用 --collect 把产物镜像回项目目录 benchmarks/re_separation/（前台执行）。

用法：
  python benchmarks/re_separation.py --run                       # 后台跑四档 → %TEMP%
  python benchmarks/re_separation.py                             # 只汇总 %TEMP% 已有结果
  python benchmarks/re_separation.py --collect                   # 前台把产物拷回项目目录
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = Path(os.path.expandvars(r"%TEMP%\re_sep"))   # 计算输出（后台可写）
OUT = ROOT / "benchmarks" / "re_separation"            # 项目归档（仅前台 collect 可写）
CASES = [
    {"d": 20, "re": 80.0},
    {"d": 20, "re": 120.0},
    {"d": 40, "re": 80.0},
    {"d": 40, "re": 120.0},
]
REF = {"Cl_amp": 2.003}  # Schäfer-Turek DFG 2D-2 的 Cl 全幅——🔴 只锚定 Re=100，对 Re=80/120 不适用（23 号指导第二节）
# 已有 Re=100 档（benchmarks/grid/*/measured.json 实测，Cl 全幅绝对值）
EXISTING = {"D20": {"tau": 0.56, "cl_amp": 1.8232},
            "D40": {"tau": 0.62, "cl_amp": 1.7180}}


def expected_tau(d: float, u: float, re: float) -> float:
    nu = u * d / re
    return 3.0 * nu + 0.5


def run_all(total_seconds: float) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for c in CASES:
        name = f"D{c['d']}_re{int(c['re'])}"
        out = RUN_DIR / name
        cmd = [sys.executable, str(ROOT / "run_cylinder.py"),
               "--d-lattice", str(c["d"]), "--re", str(c["re"]), "--u-mean-l", "0.1",
               "--total-seconds", str(total_seconds), "--run-seconds", str(total_seconds),
               "--progress-every", "20000", "--ckpt-every", "10000",
               "--out", str(out)]
        print(f"=== {name}（τ 预期 {expected_tau(c['d'], 0.1, c['re']):.4f}）=== ", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"🔴 {name} 返回 {r.returncode}，停")
            return r.returncode
    return 0


def collect() -> int:
    """把 %TEMP% 的产物镜像回项目目录（须在前台跑）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in CASES:
        src = RUN_DIR / f"D{c['d']}_re{int(c['re'])}"
        if not src.exists():
            continue
        dst = OUT / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("measured.json", "history.npz", "ckpt.npz",
                  "mass_history.png", "stationarity.png", "cl_history.png", "vorticity.png"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
                n += 1
    print(f"已镜像 {n} 个文件 → {OUT}")
    return 0


def summarize() -> int:
    rows = []
    for c in CASES:
        p = RUN_DIR / f"D{c['d']}_re{int(c['re'])}" / "measured.json"
        if not p.exists():
            p2 = OUT / f"D{c['d']}_re{int(c['re'])}" / "measured.json"  # collect 后从项目目录读
            if not p2.exists():
                print(f"缺 D{c['d']}_re{int(c['re'])}，跳过")
                continue
            p = p2
        d = json.loads(p.read_text(encoding="utf-8"))
        res, cfg = d["result"], d["config"]
        gates = res["gates"]
        if not (gates["mass_ok"] and gates["stationarity_ok"]):
            rows.append({"name": f"D{c['d']}_re{int(c['re'])}", "tau": cfg["tau"],
                         "d": c["d"], "re": c["re"], "gates_ok": False})
            continue
        m = res["steady_metrics"]
        cl = m["Cl_amp"]
        rows.append({"name": f"D{c['d']}_re{int(c['re'])}", "tau": cfg["tau"], "d": c["d"],
                     "re": c["re"], "gates_ok": True, "cl_amp": cl})

    # 🔴 口径（23 号指导第二节）：REF=2.003 只锚定 Re=100；Re=80/120 下 Cl 真值本身会随 Re 变，
    # 用它算 cl_err 会混入 ∓5% 系统污染（与要测的 5~14% 信号同量级）。
    # ✅ 正确口径：**同一 Re 下 D40 与 D20 的偏差之差**——同 Re ⇒ 真值相同 ⇒ Re 影响在相减时消掉。
    by_key = {r["name"]: r for r in rows if r["gates_ok"]}
    for r in rows:
        if not r["gates_ok"]:
            continue
        # 仅 Re=100 档有文献基准，可算绝对偏差（只作参考行）
        if r["re"] == 100.0:
            r["cl_err_re100_ref"] = abs(r["cl_amp"] - REF["Cl_amp"]) / REF["Cl_amp"]
    pairs = []
    for re_v in (80.0, 100.0, 120.0):
        r20 = by_key.get(f"D20_re{int(re_v)}") or (
            {"cl_amp": EXISTING["D20"]["cl_amp"], "tau": EXISTING["D20"]["tau"], "gates_ok": True}
            if re_v == 100.0 else None)
        r40 = by_key.get(f"D40_re{int(re_v)}") or (
            {"cl_amp": EXISTING["D40"]["cl_amp"], "tau": EXISTING["D40"]["tau"], "gates_ok": True}
            if re_v == 100.0 else None)
        if r20 and r40:
            delta = r40["cl_amp"] - r20["cl_amp"]
            pairs.append({"re": re_v, "tau20": r20["tau"], "tau40": r40["tau"],
                          "cl_d20": r20["cl_amp"], "cl_d40": r40["cl_amp"],
                          "delta_cl": delta,
                          "delta_rel": abs(delta) / r20["cl_amp"]})  # 交叉量：同 Re 两网格偏差之比（相对形式）

    L = ["# 单变量 τ 分离：固定 D 与 U=0.1，只改 Re（2026-10-28）", "",
         "> 🔴 口径（23 号指导）：**Schäfer-Turek 基准只锚定 Re=100**，对 Re=80/120 的档位",
         "> 用它算偏差会混入 ∓5% 系统污染（Cl 真值本身随 Re 变，量级与要测的信号相当）。",
         "> ✅ **主口径 = 同 Re 下 D40 − D20 的 Cl 差**：同 Re ⇒ 真值相同 ⇒ Re 的影响在相减时消掉，",
         "> 这一项就是「纯分辨率贡献」。Re=100 档另有文献基准，绝对偏差只作参考。",
         "> 门禁照旧三道，未过的档不作数。", "",
         "| 档 | D | Re | τ | 门禁 | Cl 全幅 | vs 文献（仅 Re=100 参考行） |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["gates_ok"]:
            ref_col = f"{r['cl_err_re100_ref']*100:.2f}%" if "cl_err_re100_ref" in r else "不适用（基准锚定 Re=100）"
            L.append(f"| {r['name']} | {r['d']} | {int(r['re'])} | {r['tau']:.4f} | ✅ | "
                     f"{r['cl_amp']:.4f} | {ref_col} |")
        else:
            L.append(f"| {r['name']} | {r['d']} | {int(r['re'])} | {r['tau']:.4f} | ❌ | — | — |")

    L += ["", "## 主口径：同 Re 下 D40 − D20（纯分辨率贡献；事实，不含主因判断）", "",
          "| Re | τ(D20) | τ(D40) | Cl(D20) | Cl(D40) | ΔCl = D40−D20 | 相对差（Δ/Cl(D20)） |",
          "|---|---|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| {int(p['re'])} | {p['tau20']:.4f} | {p['tau40']:.4f} | "
                 f"{p['cl_d20']:.4f} | {p['cl_d40']:.4f} | {p['delta_cl']:+.4f} | {p['delta_rel']*100:.2f}% |")

    # 零成本自检：Re=80 的 D20 vs Re=100 的 D20，Cl 绝对值直接比
    selfchk = ""
    r80 = by_key.get("D20_re80")
    if r80:
        d = (r80["cl_amp"] - EXISTING["D20"]["cl_amp"]) / EXISTING["D20"]["cl_amp"] * 100
        selfchk = (f"\n## 零成本自检（污染是否真实存在）\n\n"
                   f"- Re=80 D20 的 Cl 全幅 {r80['cl_amp']:.4f} vs Re=100 D20 的 {EXISTING['D20']['cl_amp']:.4f}"
                   f" → 差 {d:+.2f}%\n"
                   f"- {'🔴 超 5% → Re=100 基准对其他 Re 确实不适用，差值口径必须用' if abs(d) >= 5 else ('🟡 1~5% 之间 → 污染存在但偏小，仍用差值口径（更稳）' if abs(d) >= 1 else '✅ 1% 以内 → 污染可忽略，但差值口径仍更干净')}\n")
    if selfchk:
        L.append(selfchk)
    L += ["", "> 差值随 Re 怎么变 → 主因判断，等四档全过门禁后由人写进 README（生成器不下结论）。"]
    if OUT.parent.exists() and os.access(OUT.parent, os.W_OK):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "re_separation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        (OUT / "re_separation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写 {OUT/'re_separation.md'} · re_separation.json")
    else:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "re_separation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        (RUN_DIR / "re_separation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"项目目录不可写（后台），已写 {RUN_DIR/'re_separation.md'}")
    print("\n".join(L[10:]))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="后台跑四档（输出 %TEMP%）")
    ap.add_argument("--collect", action="store_true", help="前台把产物镜像回项目目录")
    ap.add_argument("--total-seconds", type=float, default=30.0)
    a = ap.parse_args()
    if a.collect:
        raise SystemExit(collect())
    if a.run:
        rc = run_all(a.total_seconds)
        if rc != 0:
            raise SystemExit(rc)
    raise SystemExit(summarize())
