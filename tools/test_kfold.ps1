# test_kfold.ps1 - K-Fold Cross Validation Testing Script
# config 파일의 _fold 값을 0-4로 변경하면서 5번 실행

# UTF-8 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 명령행 인수 처리
$ConfigFile = $args[0]
$CheckpointFile = ""
$WorkDir = ""
$ShowDir = ""
$Show = $false
$ShowInterval = 1
$CfgOptions = ""

# 인수 파싱
for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "-ConfigFile" { 
            $ConfigFile = $args[$i + 1]
            $i++
        }
        "-CheckpointFile" { 
            $CheckpointFile = $args[$i + 1]
            $i++
        }
        "-WorkDir" { 
            $WorkDir = $args[$i + 1]
            $i++
        }
        "-ShowDir" { 
            $ShowDir = $args[$i + 1]
            $i++
        }
        "-Show" { 
            $Show = $true
        }
        "-ShowInterval" { 
            $ShowInterval = [int]$args[$i + 1]
            $i++
        }
        "-CfgOptions" { 
            $CfgOptions = $args[$i + 1]
            $i++
        }
    }
}

# 매개변수 검증
if (-not $ConfigFile) {
    Write-Error "ConfigFile parameter is required. Usage: .\tools\test_kfold.ps1 -ConfigFile 'path/to/config.py' -CheckpointFile 'path/to/checkpoint.pth'"
    Write-Host "Example: .\tools\test_kfold.ps1 -ConfigFile 'configs/LSTV_cls/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4.py' -CheckpointFile 'work_dirs/fold_*/best_accuracy_top1_epoch_*.pth'" -ForegroundColor Yellow
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

# test.py 파일 경로 확인 (mmpretrain 디렉토리에서 실행하는 경우)
$testScriptPath = "tools/test.py"
if (-not (Test-Path $testScriptPath)) {
    Write-Error "test.py not found at: $testScriptPath"
    Write-Host "Please run this script from the mmpretrain directory" -ForegroundColor Yellow
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

# 0-4 fold에 대해 반복 실행
for ($fold = 0; $fold -lt 5; $fold++) {
    Write-Host "`n=== Running Test for Fold $fold ===" -ForegroundColor Cyan
    
    # config 파일에서 _fold 값 변경 (BOM 제거)
    $content = Get-Content $ConfigFile -Raw -Encoding UTF8
    # BOM 제거 (U+FEFF)
    $content = $content -replace "^\uFEFF", ""
    $content = $content -replace "_fold = \d+", "_fold = $fold"
    # UTF8NoBOM으로 저장
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($ConfigFile, $content, $utf8NoBom)
    
    Write-Host "Changed _fold value to $fold in config file." -ForegroundColor Yellow
    
    # checkpoint 파일 경로 결정
    $currentCheckpoint = $CheckpointFile
    if ($CheckpointFile -ne "") {
        # fold별로 checkpoint 파일 경로 조정 (와일드카드 처리)
        if ($CheckpointFile -match "\*") {
            # fold_* 패턴을 현재 fold로 치환
            $searchPattern = $CheckpointFile -replace "fold_\*", "fold_$fold"
            Write-Host "Searching for checkpoint with pattern: $searchPattern" -ForegroundColor Gray
            
            # 패턴에 맞는 파일 찾기 (Get-ChildItem의 -Path와 -Include 사용)
            try {
                $searchDir = Split-Path $searchPattern -Parent
                $searchFile = Split-Path $searchPattern -Leaf
                
                # 절대 경로로 변환
                if (-not [System.IO.Path]::IsPathRooted($searchDir)) {
                    $searchDir = Join-Path (Get-Location) $searchDir
                }
                
                Write-Host "Search directory: $searchDir" -ForegroundColor Gray
                Write-Host "Search file pattern: $searchFile" -ForegroundColor Gray
                
                if (Test-Path $searchDir) {
                    $matchingFiles = Get-ChildItem -Path $searchDir -Filter $searchFile -ErrorAction SilentlyContinue
                    
                    if ($matchingFiles.Count -gt 0) {
                        # 가장 최신 파일 선택 (epoch 번호가 가장 높은 것)
                        $currentCheckpoint = ($matchingFiles | Sort-Object Name -Descending)[0].FullName
                        Write-Host "Found checkpoint: $currentCheckpoint" -ForegroundColor Green
                    } else {
                        Write-Host "Warning: No checkpoint found for fold $fold in directory: $searchDir" -ForegroundColor Yellow
                        Write-Host "Looking for pattern: $searchFile" -ForegroundColor Yellow
                        continue
                    }
                } else {
                    Write-Host "Warning: Search directory does not exist: $searchDir" -ForegroundColor Yellow
                    continue
                }
            } catch {
                Write-Host "Error searching for checkpoint: $_" -ForegroundColor Red
                continue
            }
        } else {
            # 고정된 checkpoint 파일 경로인 경우 - fold 번호 치환
            $currentCheckpoint = $CheckpointFile -replace "fold_\d+", "fold_$fold"
            
            # 파일 존재 확인
            if (-not [System.IO.Path]::IsPathRooted($currentCheckpoint)) {
                $currentCheckpoint = Join-Path (Get-Location) $currentCheckpoint
            }
            
            if (-not (Test-Path $currentCheckpoint)) {
                Write-Host "Warning: Checkpoint file not found: $currentCheckpoint" -ForegroundColor Yellow
                continue
            }
        }
    }
    
    # test.py 실행 명령어 구성
    $testCmd = "$PythonCmd `"$testScriptPath`" `"$ConfigFile`""
    
    if ($currentCheckpoint -ne "") {
        $testCmd += " `"$currentCheckpoint`""
    }
    
    if ($WorkDir -ne "") {
        $testCmd += " --work-dir `"$WorkDir`""
    }
    
    if ($ShowDir -ne "") {
        $testCmd += " --show-dir `"$ShowDir`""
    }
    
    if ($Show) {
        $testCmd += " --show"
    }
    
    if ($ShowInterval -ne 1) {
        $testCmd += " --show-interval $ShowInterval"
    }
    
    if ($CfgOptions -ne "") {
        $testCmd += " --cfg-options $CfgOptions"
    }
    
    Write-Host "Command: $testCmd" -ForegroundColor Gray
    
    # test.py 실행
    try {
        Invoke-Expression $testCmd
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Test for Fold $fold completed!" -ForegroundColor Green
        } else {
            Write-Host "Error occurred in Test for Fold $fold (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "Exception occurred in Test for Fold $fold" -ForegroundColor Red
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

Write-Host "`n=== All Fold Tests Completed ===" -ForegroundColor Cyan
Write-Host "Usage (from mmpretrain directory):" -ForegroundColor Yellow
Write-Host "  .\tools\test_kfold.ps1 -ConfigFile 'configs/LSTV_cls/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4.py' -CheckpointFile '../work_dirs/lstv_classification_v2/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4/fold_*/best_accuracy_top1_epoch_*.pth'" -ForegroundColor Gray
Write-Host "  .\tools\test_kfold.ps1 -ConfigFile 'configs/LSTV_cls/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4.py' -CheckpointFile '../work_dirs/lstv_classification_v2/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4/fold_0/best_accuracy_top1_epoch_160.pth' -Show" -ForegroundColor Gray
