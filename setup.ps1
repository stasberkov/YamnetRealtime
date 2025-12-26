# setup.ps1 - Download YAMNet model and setup project
# Run: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  YAMNet Real-Time Audio Classification - Setup Script" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check .NET SDK
Write-Host "Checking .NET SDK..." -ForegroundColor Yellow
try {
    $dotnetVersion = dotnet --version
    Write-Host "  ✅ .NET SDK found: $dotnetVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ .NET SDK not found. Please install from https://dotnet.microsoft.com/download" -ForegroundColor Red
    exit 1
}

# Restore NuGet packages
Write-Host ""
Write-Host "Restoring NuGet packages..." -ForegroundColor Yellow
dotnet restore
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Packages restored successfully" -ForegroundColor Green
} else {
    Write-Host "  ❌ Failed to restore packages" -ForegroundColor Red
    exit 1
}

# Download YAMNet model
$modelDir = "yamnet_model"
$modelFile = "yamnet.tar.gz"
$modelUrl = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"

if (Test-Path $modelDir) {
    Write-Host ""
    Write-Host "Model directory already exists. Skipping download." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Downloading YAMNet model from TensorFlow Hub..." -ForegroundColor Yellow
    Write-Host "  URL: $modelUrl" -ForegroundColor Gray
    
    try {
        # Download with progress
        $ProgressPreference = 'SilentlyContinue'  # Faster download
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelFile -UseBasicParsing
        Write-Host "  ✅ Download complete" -ForegroundColor Green
        
        # Extract
        Write-Host ""
        Write-Host "Extracting model..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
        
        # Try tar (available in Windows 10+)
        tar -xzf $modelFile -C $modelDir
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Extraction complete" -ForegroundColor Green
            Remove-Item $modelFile -Force
        } else {
            Write-Host "  ❌ Extraction failed. Please extract manually:" -ForegroundColor Red
            Write-Host "     tar -xzf $modelFile -C $modelDir" -ForegroundColor Gray
        }
    } catch {
        Write-Host "  ❌ Download failed: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please download manually:" -ForegroundColor Yellow
        Write-Host "  1. Go to: https://tfhub.dev/google/yamnet/1" -ForegroundColor Gray
        Write-Host "  2. Click 'Download'" -ForegroundColor Gray
        Write-Host "  3. Extract to '$modelDir' folder" -ForegroundColor Gray
        exit 1
    }
}

# Verify model files
Write-Host ""
Write-Host "Verifying model files..." -ForegroundColor Yellow
$requiredFiles = @(
    "$modelDir/saved_model.pb"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file (missing)" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host ""
    Write-Host "Some model files are missing. Please re-download the model." -ForegroundColor Red
    exit 1
}

# Download class map
$classMapFile = "yamnet_class_map.csv"
$classMapUrl = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

if (-not (Test-Path $classMapFile)) {
    Write-Host ""
    Write-Host "Downloading class map..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri $classMapUrl -OutFile $classMapFile -UseBasicParsing
        Write-Host "  ✅ Class map downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️ Could not download class map (will be downloaded at runtime)" -ForegroundColor Yellow
    }
}

# Build project
Write-Host ""
Write-Host "Building project..." -ForegroundColor Yellow
dotnet build --configuration Release

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Build successful" -ForegroundColor Green
} else {
    Write-Host "  ❌ Build failed" -ForegroundColor Red
    exit 1
}

# Done
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Setup complete! Run the application with:" -ForegroundColor Cyan
Write-Host "    dotnet run" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
