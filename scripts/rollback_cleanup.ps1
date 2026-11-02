# Automated Rollback Script for Cleanup 20261102_202233
# Run this script in PowerShell to completely reverse all moves.
$ErrorActionPreference = 'Stop'
Write-Host 'Starting Rollback for cleanup 20261102_202233...'

if (Test-Path 'docs\historical_reports\REAL_WORLD_FAILURE_ANALYSIS.md') {
    $parent = Split-Path 'REAL_WORLD_FAILURE_ANALYSIS.md' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'docs\historical_reports\REAL_WORLD_FAILURE_ANALYSIS.md' -Destination 'REAL_WORLD_FAILURE_ANALYSIS.md' -Force
    Write-Host 'Restored: REAL_WORLD_FAILURE_ANALYSIS.md'
}
if (Test-Path 'docs\historical_reports\PROJECT_PLAN.md') {
    $parent = Split-Path 'PROJECT_PLAN.md' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'docs\historical_reports\PROJECT_PLAN.md' -Destination 'PROJECT_PLAN.md' -Force
    Write-Host 'Restored: PROJECT_PLAN.md'
}
if (Test-Path 'docs\historical_reports\MODEL_COMPARISON_REPORT.md') {
    $parent = Split-Path 'MODEL_COMPARISON_REPORT.md' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'docs\historical_reports\MODEL_COMPARISON_REPORT.md' -Destination 'MODEL_COMPARISON_REPORT.md' -Force
    Write-Host 'Restored: MODEL_COMPARISON_REPORT.md'
}
if (Test-Path 'docs\historical_reports\MODEL_CARD.md') {
    $parent = Split-Path 'MODEL_CARD.md' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'docs\historical_reports\MODEL_CARD.md' -Destination 'MODEL_CARD.md' -Force
    Write-Host 'Restored: MODEL_CARD.md'
}
if (Test-Path 'docs\historical_reports\FINAL_FRONTEND_INTEGRATION_REPORT.md') {
    $parent = Split-Path 'FINAL_FRONTEND_INTEGRATION_REPORT.md' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'docs\historical_reports\FINAL_FRONTEND_INTEGRATION_REPORT.md' -Destination 'FINAL_FRONTEND_INTEGRATION_REPORT.md' -Force
    Write-Host 'Restored: FINAL_FRONTEND_INTEGRATION_REPORT.md'
}
if (Test-Path 'docs\historical_reports\FINAL_DEMO_CHECKLIST.md') {
    $parent = Split-Path 'FINAL_DEMO_CHECKLIST.md' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'docs\historical_reports\FINAL_DEMO_CHECKLIST.md' -Destination 'FINAL_DEMO_CHECKLIST.md' -Force
    Write-Host 'Restored: FINAL_DEMO_CHECKLIST.md'
}
if (Test-Path 'docs\historical_reports\DATASET_AUDIT_REPORT.md') {
    $parent = Split-Path 'DATASET_AUDIT_REPORT.md' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'docs\historical_reports\DATASET_AUDIT_REPORT.md' -Destination 'DATASET_AUDIT_REPORT.md' -Force
    Write-Host 'Restored: DATASET_AUDIT_REPORT.md'
}
if (Test-Path 'backups\pre_cleanup_quarantine_20261102_202233\ecovision_train_val.zip') {
    $parent = Split-Path 'data\ecovision_train_val.zip' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'backups\pre_cleanup_quarantine_20261102_202233\ecovision_train_val.zip' -Destination 'data\ecovision_train_val.zip' -Force
    Write-Host 'Restored: data\ecovision_train_val.zip'
}
if (Test-Path 'backups\pre_cleanup_quarantine_20261102_202233\ecovision_bbox_v2.zip') {
    $parent = Split-Path 'ecovision_bbox_v2.zip' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'backups\pre_cleanup_quarantine_20261102_202233\ecovision_bbox_v2.zip' -Destination 'ecovision_bbox_v2.zip' -Force
    Write-Host 'Restored: ecovision_bbox_v2.zip'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch70.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch70.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch70.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch70.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch70.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch65.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch65.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch65.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch65.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch65.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch60.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch60.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch60.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch60.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch60.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch55.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch55.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch55.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch55.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch55.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch50.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch50.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch50.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch50.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch50.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch5.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch5.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch5.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch5.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch5.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch45.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch45.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch45.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch45.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch45.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch40.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch40.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch40.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch40.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch40.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch35.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch35.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch35.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch35.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch35.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch30.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch30.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch30.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch30.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch30.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch25.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch25.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch25.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch25.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch25.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch20.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch20.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch20.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch20.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch20.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch15.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch15.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch15.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch15.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch15.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch10.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch10.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch10.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch10.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch10.pt'
}
if (Test-Path 'archive\models\yolo11m_v1_epochs\epoch0.pt') {
    $parent = Split-Path 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch0.pt' -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Move-Item -Path 'archive\models\yolo11m_v1_epochs\epoch0.pt' -Destination 'models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch0.pt' -Force
    Write-Host 'Restored: models\experiments\yolo11m_v1\runs\train\yolo11m_waste_960\weights\epoch0.pt'
}
Write-Host 'Rollback complete. All items restored to exact original locations.'