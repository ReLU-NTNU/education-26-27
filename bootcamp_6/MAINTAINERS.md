# Maintaining Bootcamp 6

Student setup and notebook instructions belong in [README.md](README.md).
Keep implementation details, validation notes and platform limitations here.

## Environment and launchers

Python 3.13 and dependencies are locked in `uv.lock`. Run `uv lock` after changing
`pyproject.toml`, and keep inline notebook metadata consistent. Each bootcamp has
its own `.venv`; launchers override `UV_PROJECT_ENVIRONMENT` for the subprocess
so another active project does not receive these dependencies.

The POSIX launcher supports Intel/Apple Silicon macOS and glibc Linux. The
Windows launchers request x64 Python because the pinned `loro` dependency lacks
native Windows ARM wheels. Windows 11 ARM therefore requires x64 emulation.
Binary-only resolution has passed for Windows x64, both Mac architectures, and
x64/ARM Linux with glibc 2.28. Alpine/musl and 32-bit platforms are not tested.

The `.cmd` entry point avoids PowerShell execution-policy requirements. Scripts
resolve their own project path, preserve quoted arguments, and return uv's exit
status. They do not automatically install uv itself.

## VS Code

`bootcamp_6.code-workspace` opens this folder as the workspace root. Folder-level
settings use `${workspaceFolder}/.venv`, allowing VS Code to locate either
`bin/python` or `Scripts/python.exe`. This scopes the default to Bootcamp 6.
A previously selected interpreter or notebook kernel can override the default;
students may still need to choose the environment in the notebook kernel picker.
Marimo sandbox mode uses inline requirements in a separate environment.

## Checks

Run `sh scripts/test.sh` or `.\scripts\test.cmd` from this folder. Checks cover:

- Launching from another cwd with spaces in the project path.
- Argument forwarding, exit status and isolated Python selection.
- Blank notebook startup and complete solution execution.
- Train-only scaling, model results and preserved personal answers.

The local suite has passed on Apple Silicon macOS. The portability workflow in
`.github/workflows/bootcamp-6.yml` is manual-only (it does not run on push) and is configured for Ubuntu 22.04/24.04, Windows,
Apple Silicon macOS and Intel macOS. Other native results remain unverified until
that workflow runs; dependency resolution is not a runtime test.

Run `python ../scripts/check_repository.py` after staging. Do not commit `.venv`,
student `work/`, generated outputs or datasets. The launcher creates working
copies once; it must never overwrite an existing student's answers. To adopt a
new template, a student can rename their personal file to a backup and run setup
again.
