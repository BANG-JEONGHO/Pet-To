$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $PSScriptRoot
$labelsDir = Join-Path $rootDir "pet-images\yolo_outputs\cat2\labels"
$outputPath = Join-Path $rootDir "pet-images\yolo_outputs\cat2\labels.jsonl"

if (-not (Test-Path -LiteralPath $labelsDir)) {
    throw "Labels directory not found: $labelsDir"
}

$imageExtensions = @(".jpg", ".jpeg", ".png", ".webp")
$records = New-Object System.Collections.Generic.List[string]

Get-ChildItem -LiteralPath $labelsDir -Filter *.txt | Sort-Object Name | ForEach-Object {
    $labelFile = $_
    $imageName = $null

    foreach ($extension in $imageExtensions) {
        $candidate = Join-Path (Split-Path -Parent $labelsDir) ($labelFile.BaseName + $extension)
        if (Test-Path -LiteralPath $candidate) {
            $imageName = [System.IO.Path]::GetFileName($candidate)
            break
        }
    }

    Get-Content -LiteralPath $labelFile.FullName | ForEach-Object {
        $line = $_.Trim()
        if (-not $line) {
            return
        }

        $parts = $line -split "\s+"
        if ($parts.Count -lt 5) {
            throw "Invalid YOLO label format in $($labelFile.Name): $line"
        }

        $record = [ordered]@{
            image = $imageName
            label_file = $labelFile.Name
            class_id = [int]$parts[0]
            x_center = [double]$parts[1]
            y_center = [double]$parts[2]
            width = [double]$parts[3]
            height = [double]$parts[4]
        }

        if ($parts.Count -ge 6) {
            $record.confidence = [double]$parts[5]
        }

        $records.Add(($record | ConvertTo-Json -Compress))
    }
}

[System.IO.File]::WriteAllLines($outputPath, $records)
Write-Output "Saved JSONL: $outputPath"
