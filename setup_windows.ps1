param(
    [string]$EnvName = "em-atom-workbench",
    [string]$KernelName = "em-atom-workbench",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Project root: $projectRoot"
$envFile = Join-Path $projectRoot "environment.yml"

function Assert-LastExitCode {
    param(
        [string]$Message
    )
    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

$condaEnvJson = conda env list --json | Out-String
Assert-LastExitCode "Failed to query conda environments."
$envList = $condaEnvJson | ConvertFrom-Json
$envExists = $false
foreach ($envPath in $envList.envs) {
    if ((Split-Path $envPath -Leaf) -eq $EnvName) {
        $envExists = $true
        break
    }
}
$envReady = $false

$fallbackPackages = @(
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-image",
    "scikit-learn",
    "tifffile",
    "mrcfile",
    "h5py",
    "pyarrow",
    "pytest",
    "jupyterlab",
    "notebook",
    "ipykernel",
    "napari",
    "pyqt5",
    "hyperspy",
    "rosettasciio"
)

if ($envExists) {
    Write-Host "Updating existing conda environment: $EnvName"
    conda env update -n $EnvName -f $envFile --prune
    if ($LASTEXITCODE -eq 0) {
        $envReady = $true
    } else {
        Write-Warning "conda env update failed; falling back to a staged pip-based install."
    }
} else {
    Write-Host "Creating conda environment: $EnvName"
    conda env create -f $envFile -n $EnvName
    if ($LASTEXITCODE -eq 0) {
        $envReady = $true
    } else {
        Write-Warning "conda env create failed; falling back to a staged pip-based install."
    }
}

if (-not $envReady) {
    if (-not $envExists) {
        Write-Host "Creating minimal conda environment for fallback installation"
        conda create -n $EnvName python=$PythonVersion pip -y
        Assert-LastExitCode "Failed to create the fallback conda environment $EnvName."
    }

    Write-Host "Installing fallback package set with pip"
    conda run -n $EnvName python -m pip install @fallbackPackages
    Assert-LastExitCode "Failed to install fallback packages into $EnvName."
}

Write-Host "Installing package in editable mode"
conda run -n $EnvName python -m pip install --editable $projectRoot
Assert-LastExitCode "Failed to install the package in editable mode."

Write-Host "Registering Jupyter kernel"
conda run -n $EnvName python -m ipykernel install --user --name $KernelName --display-name "Python ($KernelName)"
Assert-LastExitCode "Failed to register the Jupyter kernel."

Write-Host "Setup complete."
Write-Host "To activate the environment, run:"
Write-Host "  conda activate $EnvName"
