[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$PythonExe = "python",

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$CondaExe = "conda",

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$WorkRoot = "$env:LOCALAPPDATA\std-nrg03",

    [Parameter()]
    [string]$OutputDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SourceUrl = "https://files.pythonhosted.org/packages/de/36/57e57957b5c66d9c86021bf3b26f6b3c6a9d9810af55997ee11b7f2603cb/nrgboost-0.0.3.tar.gz"
$SourceSha256 = "7b9e6a2a951755a75f34f1ec1185e82c4038938de6d126b046d46ce0624bbda0"
$ExpectedWheelName = "nrgboost-0.0.3-cp311-cp311-win_amd64.whl"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ToolchainLock = Join-Path $PSScriptRoot "nrgboost-windows-toolchain.explicit.txt"
$RuntimeRequirements = Join-Path $RepoRoot "requirements-nrgboost-validation.txt"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter()][string]$WorkingDirectory = ""
    )

    if ($WorkingDirectory) {
        Push-Location -LiteralPath $WorkingDirectory
    }
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Assert-AsciiPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($Path.ToCharArray() | Where-Object { [int]$_ -gt 127 } | Select-Object -First 1) {
        throw "Native Windows build paths must contain ASCII characters only: $Path"
    }
}

if ($env:OS -ne "Windows_NT") {
    throw "This diagnostic build procedure supports Windows only."
}

$WorkRoot = [System.IO.Path]::GetFullPath($WorkRoot)
Assert-AsciiPath -Path $WorkRoot
if ($WorkRoot.Length -gt 60) {
    throw "WorkRoot must be at most 60 characters because the legacy MinGW backend is not long-path safe: $WorkRoot"
}
if (Test-Path -LiteralPath $WorkRoot) {
    $existing = Get-ChildItem -LiteralPath $WorkRoot -Force | Select-Object -First 1
    if ($null -ne $existing) {
        throw "WorkRoot must be absent or empty; refusing to overwrite build state: $WorkRoot"
    }
}
else {
    New-Item -ItemType Directory -Path $WorkRoot | Out-Null
}

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $WorkRoot "output"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

foreach ($requiredFile in @($ToolchainLock, $RuntimeRequirements)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required repository file is missing: $requiredFile"
    }
}

$identity = & $PythonExe -c "import json,platform,sys; print(json.dumps({'version': list(sys.version_info[:3]), 'bits': platform.architecture()[0], 'executable': sys.executable, 'base_prefix': sys.base_prefix}))"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect Python executable: $PythonExe"
}
$pythonIdentity = $identity | ConvertFrom-Json
if (($pythonIdentity.version[0] -ne 3) -or ($pythonIdentity.version[1] -ne 11)) {
    throw "NRGBoost Windows validation requires CPython 3.11; found $($pythonIdentity.version -join '.')."
}
if ($pythonIdentity.bits -ne "64bit") {
    throw "NRGBoost Windows validation requires 64-bit CPython."
}

$archive = Join-Path $WorkRoot "nrgboost-0.0.3.tar.gz"
$sourceParent = Join-Path $WorkRoot "source"
$sourceRoot = Join-Path $sourceParent "nrgboost-0.0.3"
$toolchain = Join-Path $WorkRoot "toolchain"
$buildVenv = Join-Path $WorkRoot "build-venv"
$verifyVenv = Join-Path $WorkRoot "verify-venv"
$unrepairedDirectory = Join-Path $sourceRoot "dist"
$repairedDirectory = Join-Path $WorkRoot "repaired"

Write-Host "Downloading checksum-pinned official NRGBoost source distribution..."
Invoke-WebRequest -Uri $SourceUrl -OutFile $archive
$actualSourceSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSourceSha256 -ne $SourceSha256) {
    throw "Official source digest mismatch: expected $SourceSha256, found $actualSourceSha256"
}

New-Item -ItemType Directory -Path $sourceParent | Out-Null
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "tarfile", "-e", $archive, $sourceParent)
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "The verified archive did not contain the expected nrgboost-0.0.3 root."
}

Write-Host "Creating the checksum-locked MinGW-w64 toolchain..."
Invoke-Checked -FilePath $CondaExe -Arguments @("create", "--yes", "--prefix", $toolchain, "--file", $ToolchainLock)
$mingwBin = Join-Path $toolchain "Library\mingw-w64\bin"
$gcc = Join-Path $mingwBin "gcc.exe"
$gxx = Join-Path $mingwBin "g++.exe"
$gendef = Join-Path $mingwBin "gendef.exe"
$dlltool = Join-Path $mingwBin "dlltool.exe"
foreach ($requiredTool in @($gcc, $gxx, $gendef, $dlltool)) {
    if (-not (Test-Path -LiteralPath $requiredTool -PathType Leaf)) {
        throw "Locked toolchain is incomplete: $requiredTool"
    }
}

Write-Host "Creating isolated build and verification environments..."
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "venv", $buildVenv)
Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "venv", $verifyVenv)
$buildPython = Join-Path $buildVenv "Scripts\python.exe"
$verifyPython = Join-Path $verifyVenv "Scripts\python.exe"
$buildPackages = @(
    "setuptools==84.0.0",
    "wheel==0.48.0",
    "cffi==1.17.1",
    "pycparser==2.22",
    "delvewheel==1.13.0",
    "packaging==26.3",
    "pefile==2024.8.26"
)
Invoke-Checked -FilePath $buildPython -Arguments (@("-m", "pip", "install", "--disable-pip-version-check") + $buildPackages)

# CPython ships an MSVC import library on Windows. Distutils' MinGW backend
# needs the equivalent GNU import library; derive it from the installed,
# unmodified CPython DLL rather than altering NRGBoost source.
$pythonDll = Join-Path ([string]$pythonIdentity.base_prefix) "python311.dll"
if (-not (Test-Path -LiteralPath $pythonDll -PathType Leaf)) {
    throw "CPython runtime DLL was not found: $pythonDll"
}
$buildLibs = Join-Path $buildVenv "libs"
New-Item -ItemType Directory -Path $buildLibs -Force | Out-Null
Invoke-Checked -FilePath $gendef -Arguments @($pythonDll) -WorkingDirectory $buildLibs
$definitionFile = Join-Path $buildLibs "python311.def"
$importLibrary = Join-Path $buildLibs "libpython311.a"
Invoke-Checked -FilePath $dlltool -Arguments @("--dllname", "python311.dll", "--def", $definitionFile, "--output-lib", $importLibrary)
if (-not (Test-Path -LiteralPath $importLibrary -PathType Leaf)) {
    throw "GNU CPython import library was not generated."
}

$oldPath = $env:PATH
$oldCc = $env:CC
$oldCxx = $env:CXX
try {
    $env:PATH = "$mingwBin;$(Join-Path $toolchain 'Library\bin');$oldPath"
    # setuptools uses POSIX shlex to parse these variables. Basenames avoid
    # backslash loss; the exact binaries are resolved through the scoped PATH.
    $env:CC = "gcc"
    $env:CXX = "g++"
    Write-Host "Building the unmodified, checksum-pinned source with MinGW-w64..."
    Invoke-Checked -FilePath $buildPython -Arguments @("setup.py", "build_ext", "--compiler=mingw32", "bdist_wheel") -WorkingDirectory $sourceRoot

    $unrepairedWheel = Join-Path $unrepairedDirectory $ExpectedWheelName
    if (-not (Test-Path -LiteralPath $unrepairedWheel -PathType Leaf)) {
        throw "Build did not produce the expected CPython 3.11 wheel: $ExpectedWheelName"
    }
    New-Item -ItemType Directory -Path $repairedDirectory | Out-Null
    Invoke-Checked -FilePath $buildPython -Arguments @(
        "-m", "delvewheel", "repair",
        "--add-path", $mingwBin,
        "--wheel-dir", $repairedDirectory,
        $unrepairedWheel
    )
}
finally {
    $env:PATH = $oldPath
    $env:CC = $oldCc
    $env:CXX = $oldCxx
}

$repairedWheel = Join-Path $repairedDirectory $ExpectedWheelName
if (-not (Test-Path -LiteralPath $repairedWheel -PathType Leaf)) {
    throw "delvewheel did not produce the expected repaired wheel."
}
$finalWheel = Join-Path $OutputDirectory $ExpectedWheelName
if (Test-Path -LiteralPath $finalWheel) {
    throw "Refusing to overwrite an existing output wheel: $finalWheel"
}
Copy-Item -LiteralPath $repairedWheel -Destination $finalWheel

Write-Host "Verifying the repaired wheel in a clean environment..."
Invoke-Checked -FilePath $verifyPython -Arguments @("-m", "pip", "install", "--disable-pip-version-check", "-r", $RuntimeRequirements)
Invoke-Checked -FilePath $verifyPython -Arguments @("-m", "pip", "install", "--disable-pip-version-check", "--no-deps", $finalWheel)
Invoke-Checked -FilePath $verifyPython -Arguments @("-m", "pip", "check")
Invoke-Checked -FilePath $verifyPython -Arguments @(
    "-c",
    "import importlib.metadata as m; import nrgboost, _eval; from nrgboost.tree.eval import sample; assert m.version('nrgboost') == '0.0.3'; assert sample(4, bits=64, seed=17).shape == (4,); print('NRGBoost extension smoke check passed')"
)

$wheelSha256 = (Get-FileHash -LiteralPath $finalWheel -Algorithm SHA256).Hash.ToLowerInvariant()
$toolchainLockSha256 = (Get-FileHash -LiteralPath $ToolchainLock -Algorithm SHA256).Hash.ToLowerInvariant()
$gccVersion = (& $gcc --version | Select-Object -First 1)
$condaVersion = (& $CondaExe --version)
$buildFreeze = @(& $buildPython -m pip freeze --all)
$verifyFreeze = @(& $verifyPython -m pip freeze --all)
$provenance = [ordered]@{
    schema_version = 1
    claim = "diagnostic-windows-source-build-not-authoritative-native-parity"
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    source = [ordered]@{
        project = "nrgboost"
        version = "0.0.3"
        url = $SourceUrl
        sha256 = $SourceSha256
        source_code_modified = $false
    }
    runtime = [ordered]@{
        python = ($pythonIdentity.version -join ".")
        architecture = $pythonIdentity.bits
        platform = [System.Environment]::OSVersion.VersionString
    }
    toolchain = [ordered]@{
        conda = [string]$condaVersion
        lock_file = "tools/nrgboost-windows-toolchain.explicit.txt"
        lock_sha256 = $toolchainLockSha256
        gcc = [string]$gccVersion
        compiler = "mingw32"
        openmp_runtime_bundled_by = "delvewheel==1.13.0"
    }
    build_environment = $buildFreeze
    verification_environment = $verifyFreeze
    output = [ordered]@{
        filename = $ExpectedWheelName
        sha256 = $wheelSha256
        pip_check = "passed"
        extension_smoke_check = "passed"
    }
    authoritative_parity_evidence = "docs/evidence/nrgboost/native-parity-run-30922326384.json"
}
$provenancePath = Join-Path $OutputDirectory "nrgboost-0.0.3-windows-build-provenance.json"
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $provenancePath -Encoding utf8

Write-Host "Verified wheel: $finalWheel"
Write-Host "Wheel SHA-256: $wheelSha256"
Write-Host "Provenance: $provenancePath"
