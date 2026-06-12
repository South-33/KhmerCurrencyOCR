// Silent Mem Reduct-style memory cleanup helper for CashSnap RunLong.
//
// This intentionally does not use the Mem Reduct GUI or scheduled task because
// the stock `memreduct.exe -clean` command shows a user notification.
//
// The cleanup calls mirror the memory-list portion of Henry++ Mem Reduct's
// open-source cleanup path:
//   https://github.com/henrypp/memreduct
//
// Build on this Windows workstation with:
//   gcc -O2 -municode -Wall -Wextra -o .cache_runtime/tools/silent_mem_reduct/silent_mem_reduct.exe tools/silent_mem_reduct/silent_mem_reduct.c

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

typedef LONG NTSTATUS;
typedef NTSTATUS (NTAPI *NtSetSystemInformationFn)(
    int SystemInformationClass,
    PVOID SystemInformation,
    ULONG SystemInformationLength
);

enum {
    SystemMemoryListInformation = 80
};

typedef enum _SYSTEM_MEMORY_LIST_COMMAND {
    MemoryCaptureAccessedBits = 0,
    MemoryCaptureAndResetAccessedBits = 1,
    MemoryEmptyWorkingSets = 2,
    MemoryFlushModifiedList = 3,
    MemoryPurgeStandbyList = 4,
    MemoryPurgeLowPriorityStandbyList = 5
} SYSTEM_MEMORY_LIST_COMMAND;

static ULONGLONG available_mb(void) {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        return 0;
    }
    return status.ullAvailPhys / (1024ULL * 1024ULL);
}

static int nt_failed(NTSTATUS status) {
    return status < 0;
}

static NTSTATUS run_memory_command(NtSetSystemInformationFn fn, SYSTEM_MEMORY_LIST_COMMAND command) {
    return fn(SystemMemoryListInformation, &command, sizeof(command));
}

static const wchar_t *command_name(SYSTEM_MEMORY_LIST_COMMAND command) {
    switch (command) {
        case MemoryEmptyWorkingSets:
            return L"working_sets";
        case MemoryFlushModifiedList:
            return L"modified_list";
        case MemoryPurgeStandbyList:
            return L"standby_list";
        case MemoryPurgeLowPriorityStandbyList:
            return L"low_priority_standby_list";
        default:
            return L"unknown";
    }
}

static int has_arg(int argc, wchar_t **argv, const wchar_t *name) {
    for (int i = 1; i < argc; i++) {
        if (_wcsicmp(argv[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

static int enable_privilege(const wchar_t *name, int quiet) {
    HANDLE token = NULL;
    TOKEN_PRIVILEGES privileges;
    LUID luid;
    DWORD error;

    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
        if (!quiet) {
            fwprintf(stderr, L"silent_mem_reduct: OpenProcessToken failed gle=%lu\n", GetLastError());
        }
        return 0;
    }

    if (!LookupPrivilegeValueW(NULL, name, &luid)) {
        if (!quiet) {
            fwprintf(stderr, L"silent_mem_reduct: LookupPrivilegeValueW(%ls) failed gle=%lu\n", name, GetLastError());
        }
        CloseHandle(token);
        return 0;
    }

    privileges.PrivilegeCount = 1;
    privileges.Privileges[0].Luid = luid;
    privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    if (!AdjustTokenPrivileges(token, FALSE, &privileges, sizeof(privileges), NULL, NULL)) {
        if (!quiet) {
            fwprintf(stderr, L"silent_mem_reduct: AdjustTokenPrivileges(%ls) failed gle=%lu\n", name, GetLastError());
        }
        CloseHandle(token);
        return 0;
    }

    error = GetLastError();
    CloseHandle(token);
    if (error == ERROR_NOT_ALL_ASSIGNED) {
        if (!quiet) {
            fwprintf(stderr, L"silent_mem_reduct: privilege not assigned: %ls\n", name);
        }
        return 0;
    }

    return 1;
}

int wmain(int argc, wchar_t **argv) {
    const int full = has_arg(argc, argv, L"--full");
    const int modified = has_arg(argc, argv, L"--modified");
    const int quiet = has_arg(argc, argv, L"--quiet");
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    FARPROC proc;
    NtSetSystemInformationFn nt_set_system_information;
    ULONGLONG before;
    ULONGLONG after;
    int failures = 0;

    if (!ntdll) {
        fwprintf(stderr, L"silent_mem_reduct: failed to load ntdll.dll\n");
        return 2;
    }

    proc = GetProcAddress(ntdll, "NtSetSystemInformation");
    memcpy(&nt_set_system_information, &proc, sizeof(nt_set_system_information));
    if (!nt_set_system_information) {
        fwprintf(stderr, L"silent_mem_reduct: NtSetSystemInformation unavailable\n");
        return 2;
    }

    enable_privilege(SE_PROF_SINGLE_PROCESS_NAME, quiet);
    enable_privilege(SE_INCREASE_QUOTA_NAME, quiet);

    SYSTEM_MEMORY_LIST_COMMAND commands[4];
    int count = 0;
    commands[count++] = MemoryEmptyWorkingSets;
    commands[count++] = full ? MemoryPurgeStandbyList : MemoryPurgeLowPriorityStandbyList;
    if (modified || full) {
        commands[count++] = MemoryFlushModifiedList;
    }

    before = available_mb();

    for (int i = 0; i < count; i++) {
        NTSTATUS status = run_memory_command(nt_set_system_information, commands[i]);
        if (nt_failed(status)) {
            failures++;
            fwprintf(
                stderr,
                L"silent_mem_reduct: %ls failed ntstatus=0x%08lx\n",
                command_name(commands[i]),
                (unsigned long)status
            );
        } else if (!quiet) {
            fwprintf(stdout, L"silent_mem_reduct: %ls ok\n", command_name(commands[i]));
        }
    }

    after = available_mb();
    if (!quiet) {
        fwprintf(
            stdout,
            L"silent_mem_reduct: available_mb_before=%llu after=%llu delta=%lld\n",
            before,
            after,
            (long long)(after - before)
        );
    }

    return failures ? 1 : 0;
}
