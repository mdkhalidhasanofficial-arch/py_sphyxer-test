@ECHO OFF
REM Command file for Sphinx documentation

if "%1" == "" (
	set TARGET=help
) else (
	set TARGET=%1
)

sphinx-build -M %TARGET% . _build
