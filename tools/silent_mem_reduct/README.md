# Silent Mem Reduct Helper

Use this helper for CashSnap RunLong memory cleanup when the user asks for Mem Reduct behavior without GUI notifications.

Do not use the installed `memreductTask` scheduled task for RunLong. It launches the stock Mem Reduct GUI cleanup path and shows a user notification.

`silent_mem_reduct.c` is a small console helper based on the memory-list cleanup calls used by the open-source Henry++ Mem Reduct project: https://github.com/henrypp/memreduct. It does not create a tray icon, toast, or message box.

Build:

```powershell
New-Item -ItemType Directory -Force .cache_runtime\tools\silent_mem_reduct
gcc -O2 -municode -Wall -Wextra -o .cache_runtime\tools\silent_mem_reduct\silent_mem_reduct.exe tools\silent_mem_reduct\silent_mem_reduct.c
```

Direct execution needs elevated memory privileges. On a normal shell it is expected to fail with `STATUS_PRIVILEGE_NOT_HELD`.

Install the silent elevated task from an elevated PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\install_silent_memreduct_task.ps1
```

After that, `scripts/run_with_headroom.py --memory-clean-preset memreduct` targets `CashSnapSilentMemReduct`, not the notification-producing `memreductTask`.

Until `CashSnapSilentMemReduct` is installed, prefer `--memory-clean-preset winmemorycleaner-task` for silent cleanup because `CashSnapWinMemoryCleaner` is already installed and does not show the Mem Reduct notification.
