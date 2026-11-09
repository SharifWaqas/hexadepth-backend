# HexaDepth: Upload the vetted backend source to your GitHub repo.
# This script NEVER asks for a token. Git Credential Manager handles GitHub sign-in.
$ErrorActionPreference = "Stop"
$repo = "https://github.com/SharifWaqas/hexadepth-backend.git"
$source = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $env:TEMP ("hexadepth-git-" + [guid]::NewGuid().ToString("N"))

Write-Host "HexaDepth backend publisher" -ForegroundColor Cyan
Write-Host "Repository: $repo"
Write-Host "This will commit the extracted backend files to the main branch."

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git for Windows was not found. Please install Git from https://git-scm.com/downloads/win and rerun." -ForegroundColor Red
    exit 1
}

try {
    & git clone --depth 1 $repo $target
    if ($LASTEXITCODE -ne 0) { throw "Unable to clone. Check your GitHub sign-in and network connection." }

    Get-ChildItem -LiteralPath $source -Force | Where-Object {
        $_.Name -notin @('.git', '.gitignore', '.pytest_cache', '__pycache__', 'samples')
    } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $target $_.Name) -Recurse -Force
    }

    # Preserve GitHub's Python .gitignore and append project-specific exclusions.
    Add-Content -LiteralPath (Join-Path $target '.gitignore') -Value @'

# HexaDepth-specific ignored files
/samples/
data/real/
models/*.pt
*.weights
'@

    # Keep your private email out of public commit metadata.
    & git -C $target config user.name "Muhammad Sharif"
    & git -C $target config user.email "258407300+SharifWaqas@users.noreply.github.com"
    & git -C $target add -A
    if ($LASTEXITCODE -ne 0) { throw "Git staging failed." }

    & git -C $target diff --cached --quiet
    $changed = $LASTEXITCODE -ne 0
    if (-not $changed) {
        Write-Host "No changes to upload; your repository is already current." -ForegroundColor Green
        exit 0
    }
    Write-Host "Files staged. Publishing the backend..." -ForegroundColor Yellow
    & git -C $target commit -m "feat: publish HexaDepth backend, APIs and tests"
    if ($LASTEXITCODE -ne 0) { throw "Git commit failed." }
    & git -C $target push origin main
    if ($LASTEXITCODE -ne 0) { throw "Git push failed. Verify GitHub sign-in and repository access." }

    Write-Host "SUCCESS: Backend uploaded to GitHub!" -ForegroundColor Green
    Write-Host "https://github.com/SharifWaqas/hexadepth-backend"
}
catch {
    Write-Host ("FAILED: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host "The repository was not changed if the push failed."
    exit 1
}
finally {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
    }
}
