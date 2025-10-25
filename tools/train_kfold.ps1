# train_kfold.ps1 - K-Fold Cross Validation Training Script
# config 파일의 _fold 값을 0-4로 변경하면서 5번 실행

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
    Write-Error "ConfigFile parameter is required. Usage: .\tools\train_kfold.ps1 -ConfigFile 'path/to/config.py'"
    Write-Host "Example: .\tools\train_kfold.ps1 -ConfigFile 'configs/LSTV_cls/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4.py'" -ForegroundColor Yellow
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

# Python 실행 경로 (conda 환경 활성화)
$PythonCmd = "python"

# train.py 파일 경로 확인 (mmpretrain 디렉토리에서 실행하는 경우)
$trainScriptPath = "tools/train.py"
if (-not (Test-Path $trainScriptPath)) {
    Write-Error "train.py not found at: $trainScriptPath"
    Write-Host "Please run this script from the mmpretrain directory" -ForegroundColor Yellow
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

# 0-4 fold에 대해 반복 실행
for ($fold = 0; $fold -lt 5; $fold++) {
    Write-Host "`n=== Running Fold $fold ===" -ForegroundColor Cyan
    
    # config 파일에서 _fold 값 변경 (BOM 제거)
    $content = Get-Content $ConfigFile -Raw -Encoding UTF8
    # BOM 제거 (U+FEFF)
    $content = $content -replace "^\uFEFF", ""
    $content = $content -replace "_fold = \d+", "_fold = $fold"
    # UTF8NoBOM으로 저장
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($ConfigFile, $content, $utf8NoBom)
    
    Write-Host "Changed _fold value to $fold in config file." -ForegroundColor Yellow
    
    # train.py 실행 명령어 구성
    $trainCmd = "$PythonCmd `"$trainScriptPath`" `"$ConfigFile`""
    
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
    
    # train.py 실행
    try {
        Invoke-Expression $trainCmd
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Fold $fold completed!" -ForegroundColor Green
        } else {
            Write-Host "Error occurred in Fold $fold (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "Exception occurred in Fold $fold" -ForegroundColor Red
    }
    
    # 잠시 대기 (선택사항)
    Start-Sleep -Seconds 2
}

# 원본 config 파일 복원 (BOM 제거)
$backupContent = Get-Content $BackupFile -Raw -Encoding UTF8
$backupContent = $backupContent -replace "^\uFEFF", ""
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($ConfigFile, $backupContent, $utf8NoBom)
Write-Host "`nOriginal config file restored (BOM removed)." -ForegroundColor Green

Write-Host "`n=== All Folds Completed ===" -ForegroundColor Cyan
Write-Host "Usage (from mmpretrain directory): .\tools\train_kfold.ps1 -ConfigFile 'configs/LSTV_cls/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4.py'" -ForegroundColor Yellow
Write-Host "Example: .\tools\train_kfold.ps1 -ConfigFile 'configs/LSTV_cls/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4.py' -Amp" -ForegroundColor Yellow