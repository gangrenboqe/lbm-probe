# 收官单驱动脚本：零干预跑完全部配置（分段 + checkpoint 续跑，中断不丢）
#
# 配置（全部同口径：--total-seconds 30 --re 100 --n-sub 4；每档独立 --out 目录）
#   D20 / D30 / D40        网格收敛（主目标）—— 固定 U=0.1、Re=100，故 τ 随 D 变化（必然，见 summary.md）
#   tau055 / tau057        τ 敏感性 —— **保持 Re=100**，由 τ 反推 U（τ=3ν+0.5 ⇒ U=Re·ν/D）
#                          ⚠️ LBM 里 τ 不是自由参数：固定 U/D 只改 τ 等于改 Re，故本实验按 Re 固定口径做
#   D30_copy               D=30 出口模式对照（旧整列拷贝）
#
# 每档：3 段 × 10 物理秒 → 再 --finalize 出门禁结论；日志与 checkpoint 都落在该档目录

$ErrorActionPreference = 'Continue'
$env:PYTHONDONTWRITEBYTECODE = '1'
$root = 'E:\DSHPersonalAI\projects\lbm-probe'
$script = Join-Path $root 'run_cylinder.py'

$configs = @(
  @{ dir = 'D20';      a = @('--d-lattice', '20', '--u-mean-l', '0.1') },
  @{ dir = 'D30';      a = @('--d-lattice', '30', '--u-mean-l', '0.1') },
  @{ dir = 'D40';      a = @('--d-lattice', '40', '--u-mean-l', '0.1') },
  @{ dir = 'tau055';   a = @('--d-lattice', '20', '--u-mean-l', '0.0833333333') },
  @{ dir = 'tau057';   a = @('--d-lattice', '20', '--u-mean-l', '0.1166666667') },
  @{ dir = 'D30_copy'; a = @('--d-lattice', '30', '--u-mean-l', '0.1', '--legacy-outlet') }
)

foreach ($c in $configs) {
  $out = Join-Path $root ('benchmarks\grid\' + $c.dir)
  # 目录已由外部预建（沙箱策略：禁止在改写路径里新建目录，允许写入已存在目录）
  if (-not (Test-Path $out)) { "!! 目录不存在，跳过 $($c.dir)：$out" | Out-File -Append -Encoding utf8 (Join-Path $root 'benchmarks\grid\CAMPAIGN_DONE.txt'); continue }
  $log = Join-Path $out 'run.log'
  "=== [$($c.dir)] 开始 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -Append -Encoding utf8 $log
  for ($i = 1; $i -le 3; $i++) {
    "--- [$($c.dir)] 第 $i/3 段 $(Get-Date -Format 'HH:mm:ss') ---" | Out-File -Append -Encoding utf8 $log
    $a = $c.a + @('--run-seconds', '10', '--total-seconds', '30', '--progress-every', '5000', '--out', $out)
    & python $script @a *>&1 | Tee-Object -FilePath $log -Append
    "--- [$($c.dir)] 第 $i 段结束 exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss') ---" | Out-File -Append -Encoding utf8 $log
  }
  "--- [$($c.dir)] finalize $(Get-Date -Format 'HH:mm:ss') ---" | Out-File -Append -Encoding utf8 $log
  $b = $c.a + @('--finalize', '--out', $out)
  & python $script @b *>&1 | Tee-Object -FilePath $log -Append
  "### [$($c.dir)] 完成 $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
}

"=== 全部配置结束 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -Append -Encoding utf8 (Join-Path $root 'benchmarks\grid\CAMPAIGN_DONE.txt')
