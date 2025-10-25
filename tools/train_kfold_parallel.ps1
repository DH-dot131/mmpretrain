# train_kfold_parallel.ps1 - Parallel K-Fold Cross Validation Training Script
# config 파일의 _fold 값을 0-4로 변경하면서 5개를 동시에 병렬 실행
# ⚠️ 주의: Backbone이 freeze된 경우(FC layer만 학습)에만 사용하세요!

# UTF-8 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 명령행 인수 처리
$ConfigFile = $args[0]
$WorkDir = ""
$Resume = $false
$Amp = $false
$NoValidate = $false
$AutoScaleLr = $false
$CfgOptions = ""

# 인수 파싱
for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "-ConfigFile" { 
            $ConfigFile = $args[$i + 1]
            $i++
        }
        "-WorkDir" { 
            $WorkDir = $args[$i + 1]
            $i++
        }
        "-Resume" { 
            $Resume = $true
        }
        "-Amp" { 
            $Amp = $true
        }
        "-NoValidate" { 
            $NoValidate = $true
        }
        "-AutoScaleLr" { 
            $AutoScaleLr = $true
        }
        "-CfgOptions" { 
            $CfgOptions = $args[$i + 1]
            $i++
        }
    }
}

# 매개변수 검증
if (-not $ConfigFile) {
    Write-Error "ConfigFile parameter is required. Usage: .\tools\train_kfold_parallel.ps1 -ConfigFile 'path/to/config.py'"
    Write-Host "Example: .\tools\train_kfold_parallel.ps1 -ConfigFile 'configs/LSTV_cls/clip-vit-base-p32-frozen-adaw-lr1e-3-warm5-cos-bs64-ep50_openai-pre-fs2-d0.3.py'" -ForegroundColor Yellow
    exit 1
}

# config 파일 경로 확인 (절대 경로로 변환)
if (-not [System.IO.Path]::IsPathRooted($ConfigFile)) {
    $ConfigFile = Join-Path (Get-Location) $ConfigFile
}

Write-Host "Config file path: $ConfigFile" -ForegroundColor Gray

if (-not (Test-Path $ConfigFile)) {
    Write-Error "Config file not found: $ConfigFile"
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

# 원본 config 파일 백업
$BackupFile = $ConfigFile + ".backup"
if (-not (Test-Path $BackupFile)) {
    Copy-Item $ConfigFile $BackupFile
    Write-Host "Original config file backed up: $BackupFile" -ForegroundColor Green
}

# Python 실행 경로
$PythonCmd = "python"

# train.py 파일 경로 확인
$trainScriptPath = "tools/train.py"
if (-not (Test-Path $trainScriptPath)) {
    Write-Error "train.py not found at: $trainScriptPath"
    Write-Host "Please run this script from the mmpretrain directory" -ForegroundColor Yellow
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

# 각 fold용 임시 config 파일 생성
$TempConfigs = @()
for ($fold = 0; $fold -lt 5; $fold++) {
    $tempConfigPath = $ConfigFile -replace '\.py$', "_fold$fold.py"
    
    # config 내용 읽기
    $content = Get-Content $ConfigFile -Raw -Encoding UTF8
    $content = $content -replace "^\uFEFF", ""  # BOM 제거
    
    # _fold 값 변경
    $content = $content -replace "_fold = \d+", "_fold = $fold"
    
    # UTF8NoBOM으로 저장
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($tempConfigPath, $content, $utf8NoBom)
    
    $TempConfigs += $tempConfigPath
    Write-Host "Created temporary config for fold $fold : $tempConfigPath" -ForegroundColor Green
}

# Job 배열
$Jobs = @()

Write-Host "`n=== Starting 5 Folds in Parallel ===" -ForegroundColor Cyan

# 0-4 fold를 병렬로 시작
for ($fold = 0; $fold -lt 5; $fold++) {
    Write-Host "Starting Fold $fold in background..." -ForegroundColor Yellow
    
    $tempConfig = $TempConfigs[$fold]
    
    # train.py 실행 명령어 구성
    $trainCmd = "$PythonCmd `"$trainScriptPath`" `"$tempConfig`""
    
    if ($WorkDir -ne "") {
        $trainCmd += " --work-dir `"$WorkDir`""
    }
    
    if ($Resume) {
        $trainCmd += " --resume"
    }
    
    if ($Amp) {
        $trainCmd += " --amp"
    }
    
    if ($NoValidate) {
        $trainCmd += " --no-validate"
    }
    
    if ($AutoScaleLr) {
        $trainCmd += " --auto-scale-lr"
    }
    
    if ($CfgOptions -ne "") {
        $trainCmd += " --cfg-options $CfgOptions"
    }
    
    Write-Host "Command: $trainCmd" -ForegroundColor Gray
    
    # PowerShell Job으로 백그라운드 실행
    $job = Start-Job -ScriptBlock {
        param($cmd)
        Invoke-Expression $cmd
    } -ArgumentList $trainCmd -Name "Fold_$fold"
    
    $Jobs += $job
    Write-Host "Fold $fold started (Job ID: $($job.Id))" -ForegroundColor Green
    
    # 각 job 시작 사이에 짧은 대기 (GPU 초기화 분산)
    Start-Sleep -Seconds 3
}

Write-Host "`n=== All 5 Folds are running in parallel ===" -ForegroundColor Cyan
Write-Host "Waiting for all jobs to complete..." -ForegroundColor Yellow
Write-Host "You can monitor progress in work_dirs folders" -ForegroundColor Gray

# 모든 job이 완료될 때까지 대기 및 진행 상황 표시
while ($Jobs | Where-Object { $_.State -eq 'Running' }) {
    $runningCount = ($Jobs | Where-Object { $_.State -eq 'Running' }).Count
    $completedCount = ($Jobs | Where-Object { $_.State -eq 'Completed' }).Count
    $failedCount = ($Jobs | Where-Object { $_.State -eq 'Failed' }).Count
    
    Write-Host "`r[Progress] Running: $runningCount | Completed: $completedCount | Failed: $failedCount" -NoNewline -ForegroundColor Cyan
    Start-Sleep -Seconds 10
}

Write-Host "`n`n=== All Jobs Finished ===" -ForegroundColor Cyan

# 결과 확인
for ($fold = 0; $fold -lt 5; $fold++) {
    $job = $Jobs[$fold]
    Write-Host "`nFold $fold (Job ID: $($job.Id)):" -ForegroundColor Yellow
    
    if ($job.State -eq 'Completed') {
        Write-Host "  Status: Completed ✓" -ForegroundColor Green
    } elseif ($job.State -eq 'Failed') {
        Write-Host "  Status: Failed ✗" -ForegroundColor Red
        Write-Host "  Error:" -ForegroundColor Red
        Receive-Job -Job $job 2>&1 | Write-Host -ForegroundColor Red
    } else {
        Write-Host "  Status: $($job.State)" -ForegroundColor Yellow
    }
    
    # Job 제거
    Remove-Job -Job $job -Force
}

# 임시 config 파일 삭제
Write-Host "`nCleaning up temporary config files..." -ForegroundColor Yellow
foreach ($tempConfig in $TempConfigs) {
    if (Test-Path $tempConfig) {
        Remove-Item $tempConfig -Force
        Write-Host "Removed: $tempConfig" -ForegroundColor Gray
    }
}

Write-Host "`n=== All Folds Completed ===" -ForegroundColor Cyan
Write-Host "Usage: .\tools\train_kfold_parallel.ps1 -ConfigFile 'configs/LSTV_cls/clip-vit-base-p32-frozen-adaw-lr1e-3-warm5-cos-bs64-ep50_openai-pre-fs2-d0.3.py'" -ForegroundColor Yellow
Write-Host "Note: Use this only when backbone is frozen (FC layer only training)!" -ForegroundColor Red

