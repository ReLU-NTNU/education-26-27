@echo off
call "%~dp0run.cmd" test %*
exit /b %ERRORLEVEL%
