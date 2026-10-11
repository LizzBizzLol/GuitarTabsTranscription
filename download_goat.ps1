param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$FileName,
    [long]$ExpectedBytes = 0,
    [string]$Sha256 = '',
    [int]$IntervalSeconds = 5
)
$ErrorActionPreference = 'Stop'
if ($Url -notmatch '^https://') { throw 'An HTTPS download URL is required.' }
if ([IO.Path]::GetFileName($FileName) -ne $FileName -or $FileName -match '[<>:"/\\|?*]' -or $FileName -in @('.', '..')) {
    throw 'FileName must be a plain file name, without directory components.'
}
if ($ExpectedBytes -lt 0 -or $IntervalSeconds -lt 1) { throw 'Invalid size or interval.' }
if ($Sha256 -and $Sha256 -notmatch '^[a-fA-F0-9]{64}$') { throw 'Invalid SHA256.' }
$root = Join-Path $PSScriptRoot 'datasets\GOAT'
New-Item -ItemType Directory -Path $root -Force | Out-Null
$target = Join-Path $root $FileName
$partial = $target + '.part'
$errorLog = Join-Path $root 'download.stderr.log'
$progressLog = Join-Path $root 'download.progress.log'
if (Test-Path -LiteralPath $target) { throw 'Destination already exists; it will not be overwritten.' }
$initial = if (Test-Path -LiteralPath $partial) { (Get-Item -LiteralPath $partial).Length } else { 0 }
if ($ExpectedBytes -gt 0 -and $initial -gt $ExpectedBytes) { throw 'Partial file exceeds expected size.' }
$free = (Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0,1))).Free
if ($ExpectedBytes -gt 0 -and $free -lt ($ExpectedBytes - $initial + 5GB)) { throw 'Insufficient disk space.' }
$started = Get-Date
$previousTime = $started
$previousBytes = $initial
$success = $false
Write-Host "GOAT download | destination: $target"
Write-Host 'Progress is also saved to download.progress.log. Partial files support resume.'
for ($attempt = 1; $attempt -le 10; $attempt++) {
    # URLs are deliberately excluded from progress logs (access links may be private).
    $arguments = @('--location', '--fail', '--silent', '--show-error', '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120', '--continue-at', '-', '--output', ('"' + $partial + '"'), ('"' + $Url + '"'))
    if ($Url.Contains('"') -or $Url.Contains("`r") -or $Url.Contains("`n")) { throw 'Invalid URL characters.' }
    $worker = Start-Process -FilePath 'curl.exe' -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardError $errorLog
    while (-not $worker.HasExited) {
        $now = Get-Date
        $bytes = if (Test-Path -LiteralPath $partial) { (Get-Item -LiteralPath $partial).Length } else { 0 }
        $seconds = ($now - $previousTime).TotalSeconds
        $speed = if ($seconds -gt 0) { [Math]::Max(0, ($bytes - $previousBytes) / $seconds) } else { 0 }
        $percent = if ($ExpectedBytes -gt 0) { '{0:F2}%' -f (100 * $bytes / $ExpectedBytes) } else { 'total size unknown' }
        $eta = if ($ExpectedBytes -gt 0 -and $speed -gt 0) { [TimeSpan]::FromSeconds([Math]::Max(0, ($ExpectedBytes - $bytes) / $speed)).ToString('d\.hh\:mm\:ss') } else { 'calculating' }
        $free = (Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0,1))).Free
        $line = '{0:HH:mm:ss} | PID={1} | attempt={2}/10 | {3:F3} GiB | {4} | speed={5:F2} MiB/s | elapsed={6} | ETA={7} | free={8:F2} GiB' -f $now, $worker.Id, $attempt, ($bytes/1GB), $percent, ($speed/1MB), ($now-$started).ToString('d\.hh\:mm\:ss'), $eta, ($free/1GB)
        Write-Host $line
        Add-Content -LiteralPath $progressLog -Value $line -Encoding UTF8
        if ($free -lt 5GB) {
            Stop-Process -Id $worker.Id -ErrorAction SilentlyContinue
            throw 'Stopped download: less than 5 GiB free. Partial file preserved.'
        }
        $previousBytes = $bytes
        $previousTime = $now
        Start-Sleep -Seconds $IntervalSeconds
        $worker.Refresh()
    }
    $worker.WaitForExit()
    $worker.Refresh()
    if ($worker.ExitCode -eq 0) { $success = $true; break }
    Write-Host "Download interrupted (exit $($worker.ExitCode)); partial file preserved." -ForegroundColor Yellow
    Get-Content -LiteralPath $errorLog -Tail 4
    if ($attempt -lt 10) { Start-Sleep -Seconds 15 }
}
if (-not $success) { throw 'Download failed after retries. See download.stderr.log; rerun to resume.' }
$length = (Get-Item -LiteralPath $partial).Length
if ($length -eq 0) { throw 'Downloaded file is empty.' }
if ($ExpectedBytes -gt 0 -and $length -ne $ExpectedBytes) { throw "Size mismatch: $length bytes. Partial file preserved." }
if ($Sha256) {
    Write-Host 'Verifying SHA256...'
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $Sha256) { throw 'SHA256 mismatch. Partial file preserved.' }
}
Move-Item -LiteralPath $partial -Destination $target
$line = "Completed: $length bytes | elapsed=$((Get-Date)-$started) | size verified=$($ExpectedBytes -gt 0) | SHA256 verified=$([bool]$Sha256)"
Write-Host $line -ForegroundColor Green
Add-Content -LiteralPath $progressLog -Value $line -Encoding UTF8
