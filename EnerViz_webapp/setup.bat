@echo off
title Installing Requirements
cd /d "%~dp0"
pip install -r requirements.txt
python -m nltk.downloader stopwords

set "currentDir=%~dp0"

set "target=%currentDir%\run.bat"
set "shortcut=%userprofile%\Desktop\EnerViz.lnk"
set "icon=%currentDir%\assets\Enerviz.ico"

set "WshShell=WScript.Shell"
set "oShellLink=%temp%\Shortcut.vbs"

echo Set oWS = WScript.CreateObject("%WshShell%") > "%oShellLink%"
echo sLinkFile = "%shortcut%" >> "%oShellLink%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%oShellLink%"
echo oLink.TargetPath = "%target%" >> "%oShellLink%"
echo oLink.IconLocation = "%icon%" >> "%oShellLink%"
echo oLink.Save >> "%oShellLink%"

cscript.exe "%oShellLink%"

del "%oShellLink%"