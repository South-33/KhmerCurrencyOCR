Option Explicit

Dim shell
Dim exitCode

Set shell = CreateObject("WScript.Shell")
exitCode = shell.Run("schtasks /Run /TN CashSnapSilentMemReduct", 0, True)

WScript.Quit exitCode
