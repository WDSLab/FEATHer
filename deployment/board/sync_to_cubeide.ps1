# Copy the board sources and generated headers into the CubeIDE project.
#
# The project lives outside the repo (and outside Dropbox), so every change
# here has to be carried over by hand — which is exactly how the board ended
# up running a 453-parameter FEATHer while the paper reported 938. Run this
# after any export_golden*.py or src/ change, before building in CubeIDE.
#
#     powershell -File deployment/board/sync_to_cubeide.ps1
#     powershell -File deployment/board/sync_to_cubeide.ps1 -WhatIf   # dry run
param(
    [string]$Project = "C:\Users\lee\STM32Cube_projects\blink_test",
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$board = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $Project "Core\Src"
$inc = Join-Path $Project "Core\Inc"

foreach ($d in @($src, $inc)) {
    if (-not (Test-Path $d)) { throw "CubeIDE folder not found: $d" }
}

$copied = 0
$skipped = 0

function Sync-One($file, $dest) {
    $target = Join-Path $dest $file.Name
    $same = (Test-Path $target) -and
            ((Get-FileHash $file.FullName).Hash -eq (Get-FileHash $target).Hash)
    if ($same) {
        $script:skipped++
        return
    }
    if ($WhatIf) {
        Write-Host ("  WOULD COPY  {0,-26} -> {1}" -f $file.Name, (Split-Path -Leaf $dest))
    } else {
        Copy-Item $file.FullName $target -Force
        Write-Host ("  copied      {0,-26} -> {1}" -f $file.Name, (Split-Path -Leaf $dest))
    }
    $script:copied++
}

Write-Host "CubeIDE project: $Project`n"

Write-Host "Core\Src  (.c):"
Get-ChildItem (Join-Path $board "src") -Filter *.c | ForEach-Object { Sync-One $_ $src }

Write-Host "`nCore\Inc  (.h from src/ and golden/):"
Get-ChildItem (Join-Path $board "src") -Filter *.h | ForEach-Object { Sync-One $_ $inc }
Get-ChildItem (Join-Path $board "golden") -Filter *.h | ForEach-Object { Sync-One $_ $inc }

Write-Host "`n$copied file(s) $(if ($WhatIf) {'would be '} else {''})copied, $skipped already identical."

if (-not $WhatIf -and $copied -gt 0) {
    Write-Host "`nIn CubeIDE: Project -> Clean, then pick ONE model #define in"
    Write-Host "Core\Src\main.c and Build -> Run.  (RAM is 20 KB; FEATHer alone"
    Write-Host "uses 16.6 KB, so two models at once will overflow.)"
}
