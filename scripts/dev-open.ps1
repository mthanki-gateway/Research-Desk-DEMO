<#
.SYNOPSIS
    Open every Research Desk web UI that is currently running.

.DESCRIPTION
    Probes each published port and opens ONLY the services that answer, then
    prints a table of what was opened and what was not.

    Probing rather than opening blindly is the entire point. Most of this stack
    lives behind compose profiles -- Langfuse needs `--profile obs`, pgweb needs
    `--profile tools` -- so a script that opened all five URLs would, on a
    typical day, produce three working tabs and two connection errors. A
    connection error looks like a broken app, not like a service you chose not
    to start.

    The probe is a raw TCP connect, not an HTTP request. The question is "is
    something listening", and a TCP check answers it in milliseconds without
    caring whether the app returns 200, 302 or 404 on its root path -- Qdrant's
    root 404s, and Invoke-WebRequest treats that as a terminating error.

.PARAMETER Up
    Bring the default stack up first (docker compose up -d), wait for it, then
    open. Does not start the obs or tools profiles -- those are opt-in on
    purpose, being heavy (Langfuse pulls ClickHouse, Redis and MinIO).

.PARAMETER TimeoutMs
    Per-port probe timeout. Raise it if a cold container is missed.

.EXAMPLE
    .\dev-open.ps1
    Open whatever is already running.

.EXAMPLE
    .\dev-open.ps1 -Up
    Start the default stack, then open it.
#>
[CmdletBinding()]
param(
    [switch]$Up,
    [int]$TimeoutMs = 500
)

$ErrorActionPreference = 'Stop'

# Every published web UI in docker-compose.yml. Anything without a port
# mapping is deliberately absent: MinIO is internal to the Langfuse stack, and
# Postgres, ClickHouse and Redis are not web UIs at all -- pgweb is how you
# look at the database.
$services = @(
    [pscustomobject]@{
        Name  = 'Frontend'
        Url   = 'http://localhost:3000'
        Port  = 3000
        Start = 'docker compose up -d web'
    }
    [pscustomobject]@{
        Name = 'API docs (Swagger)'
        # /docs, not /. The root has no UI, and /docs is served only when
        # APP_ENV=dev -- which compose sets. In prod it is deliberately absent.
        Url   = 'http://localhost:8000/docs'
        Port  = 8000
        Start = 'docker compose up -d api'
    }
    [pscustomobject]@{
        Name  = 'Qdrant dashboard'
        Url   = 'http://localhost:6333/dashboard'
        Port  = 6333
        Start = 'docker compose up -d qdrant'
    }
    [pscustomobject]@{
        Name  = 'Langfuse'
        Url   = 'http://localhost:3001'
        Port  = 3001
        Start = 'docker compose --profile obs up -d'
    }
    [pscustomobject]@{
        Name  = 'pgweb (database)'
        Url   = 'http://localhost:8081'
        Port  = 8081
        Start = 'docker compose --profile tools up -d pgweb'
    }
)

function Test-Port {
    <#
        True if something is listening on localhost:$Port.

        BeginConnect + WaitOne rather than Test-NetConnection, which takes
        seconds per closed port and would make this script feel broken, and
        rather than ConnectAsync().Wait(), which wraps a refused connection in
        an AggregateException that is fiddlier to swallow cleanly.
    #>
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$Timeout = 500
    )

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        # 127.0.0.1, not 'localhost'. On Windows, localhost resolves to ::1
        # first, and a container published to 0.0.0.0 may not be listening on
        # IPv6 -- so the name form can report a running service as down.
        $handle = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $handle.AsyncWaitHandle.WaitOne($Timeout, $false)) {
            return $false
        }
        $client.EndConnect($handle)
        return $true
    } catch {
        # Connection refused is the normal "not running" answer, not an error
        # worth surfacing.
        return $false
    } finally {
        $client.Close()
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot

if ($Up) {
    Write-Host ''
    Write-Host 'Starting the default stack...' -ForegroundColor Cyan
    Push-Location $repoRoot
    try {
        docker compose up -d
        if (-not $?) {
            Write-Host ''
            Write-Host 'docker compose failed. Is Docker Desktop running?' -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
    # The api container is up before uvicorn is listening, so an immediate
    # probe would miss it and the script would report its own stack as down.
    Write-Host 'Waiting for ports to open...' -ForegroundColor DarkGray
    Start-Sleep -Seconds 3
}

$running = @()
$stopped = @()

foreach ($service in $services) {
    if (Test-Port -Port $service.Port -Timeout $TimeoutMs) {
        $running += $service
    } else {
        $stopped += $service
    }
}

Write-Host ''
if ($running.Count -eq 0) {
    Write-Host 'Nothing is running.' -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'Start the stack with:' -ForegroundColor DarkGray
    Write-Host '    .\scripts\dev-open.ps1 -Up' -ForegroundColor White
    Write-Host ''
    exit 0
}

Write-Host ("Opening {0} service(s)" -f $running.Count) -ForegroundColor Green
foreach ($service in $running) {
    Write-Host ('  {0,-20} {1}' -f $service.Name, $service.Url)
    Start-Process $service.Url
    # A cold browser needs a moment to become the target for subsequent tabs.
    # Without this the first few Start-Process calls can race the launch and
    # silently drop tabs.
    Start-Sleep -Milliseconds 400
}

if ($stopped.Count -gt 0) {
    Write-Host ''
    Write-Host 'Not running:' -ForegroundColor DarkGray
    foreach ($service in $stopped) {
        Write-Host ('  {0,-20} {1}' -f $service.Name, $service.Start) -ForegroundColor DarkGray
    }
}

Write-Host ''
