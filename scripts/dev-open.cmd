@echo off
REM Double-clickable launcher for dev-open.ps1.
REM
REM This wrapper exists because a .ps1 cannot be run by double-clicking: the
REM default shell association OPENS it in an editor, and even from a shell the
REM machine's ExecutionPolicy (RemoteSigned by default) refuses a script that
REM came from a zip or a network share. A .cmd has neither problem.
REM
REM   -NoProfile        skip the user's PowerShell profile: faster, and cannot
REM                     be broken by anything in it
REM   -ExecutionPolicy  Bypass applies to THIS process only; nothing about the
REM                     machine's policy is changed
REM   %~dp0             the directory this .cmd lives in, so it works from any
REM                     working directory
REM   %*                pass arguments through, e.g.  dev-open.cmd -Up

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev-open.ps1" %*

REM Double-clicked, the window closes the instant the script ends and any
REM message is unreadable. Pause only when there is no console attached to a
REM parent shell -- CMDCMDLINE holds the full command line of the hosting
REM cmd.exe, and a double-click always runs it with /c.
echo %CMDCMDLINE% | find /i "/c" >nul
if not errorlevel 1 pause
