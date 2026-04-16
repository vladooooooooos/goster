Option Explicit

Dim fso, shell, scriptDir, parentDir, pythonPath, scriptPath, command

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
parentDir = fso.GetParentFolderName(scriptDir)
scriptPath = fso.BuildPath(scriptDir, "run_chat_user.py")

If fso.FileExists(fso.BuildPath(scriptDir, ".venv\Scripts\pythonw.exe")) Then
    pythonPath = fso.BuildPath(scriptDir, ".venv\Scripts\pythonw.exe")
ElseIf fso.FileExists(fso.BuildPath(parentDir, ".venv\Scripts\pythonw.exe")) Then
    pythonPath = fso.BuildPath(parentDir, ".venv\Scripts\pythonw.exe")
Else
    pythonPath = "pythonw.exe"
End If

shell.CurrentDirectory = scriptDir
command = """" & pythonPath & """ """ & scriptPath & """"
shell.Run command, 0, False
