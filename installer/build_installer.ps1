# PyInstaller 按 NeuroLens.spec 将 NeuroLens.exe 与 _internal 输出到仓库根目录，再用 Inno Setup 编译安装包。
# 依赖：Python 虚拟环境 NLEnv、Inno Setup 6（ISCC.exe）
$ErrorActionPreference = "Stop"
$InstallerDir = $PSScriptRoot
$RepoRoot = Split-Path $InstallerDir -Parent
Set-Location $RepoRoot

$VersionFile = Join-Path $RepoRoot "VERSION"
if (-not (Test-Path $VersionFile)) {
  throw "未找到 $VersionFile，请在仓库根目录放置 VERSION（首行为语义化版本号，如 1.0.1）。"
}
$RawVersion = (Get-Content $VersionFile -Raw).Trim() -replace "[\r\n]+", ""
$Seg = $RawVersion -split "\."
while ($Seg.Length -lt 4) { $Seg += "0" }
$VersionInfo = ($Seg[0..3] -join ".")
$AppVersionIss = Join-Path $InstallerDir "_AppVersion.iss"
@"
#define MyAppVersion "$RawVersion"
#define MyVersionInfo "$VersionInfo"
"@ | Set-Content -Path $AppVersionIss -Encoding utf8
Write-Host "==> 版本: $RawVersion (Inno: $VersionInfo)"

$Python = Join-Path $RepoRoot "..\NLEnv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  throw "未找到 $Python，请先创建虚拟环境 NLEnv。"
}

& $Python (Join-Path $RepoRoot "scripts\sync_package_json_version.py")

Write-Host "==> PyInstaller: visualization\backend\NeuroLens.spec"
& $Python -m PyInstaller (Join-Path $RepoRoot "visualization\backend\NeuroLens.spec") -y

$RootExe = Join-Path $RepoRoot "NeuroLens.exe"
$RootInternal = Join-Path $RepoRoot "_internal"
if (-not (Test-Path $RootExe) -or -not (Test-Path $RootInternal)) {
  throw "PyInstaller 未在仓库根目录生成 NeuroLens.exe / _internal，请检查 NeuroLens.spec 的 DISTPATH。"
}
$nerModelNames = @('chinese-ner-per-addr-rbt3', 'chinese-ner-per-addr')
$nerWeight = $null
foreach ($nerName in $nerModelNames) {
  $nerWeight = Get-ChildItem -Path (Join-Path $RootInternal "Desensization-dashboard\models\$nerName") -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @('model.safetensors', 'pytorch_model.bin') } |
    Select-Object -First 1
  if ($nerWeight) { break }
}
if (-not $nerWeight) {
  throw "打包产物缺少脱敏 NER 模型权重：_internal\Desensization-dashboard\models\chinese-ner-per-addr-rbt3\model.safetensors"
}
Write-Host "==> NER 模型已打入: $($nerWeight.FullName) ($([math]::Round($nerWeight.Length / 1MB, 1)) MB)"

$rootModels = Join-Path $RepoRoot "models"
if (Test-Path $rootModels) {
  $rootModelCount = (Get-ChildItem $rootModels -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
  Write-Host "==> 根目录 models\ 已就绪: $rootModelCount 个文件（将完整打入安装包）"
} else {
  Write-Host "==> 警告: 根目录 models\ 不存在，安装包将跳过该目录"
}

$iscc = @(
  "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
  "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $iscc) {
  throw "未找到 Inno Setup 6（ISCC.exe）。请从 https://jrsoftware.org/isinfo.php 安装并勾选命令行工具。"
}

$iss = Join-Path $InstallerDir "NeuroLens_Setup.iss"
Write-Host "==> Inno Setup: $iss"
& $iscc $iss

$out = Join-Path $InstallerDir "Output"
Write-Host "完成。安装包目录: $out"
