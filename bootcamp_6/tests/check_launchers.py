"""Run the real launchers from an unrelated cwd with spaces in the project path."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to test the launchers")
    with tempfile.TemporaryDirectory(prefix="relu launcher ") as directory:
        temp = Path(directory)
        project = temp / "course with spaces" / "bootcamp_6"
        scripts = project / "scripts"
        scripts.mkdir(parents=True)
        for source in (ROOT / "scripts").iterdir():
            if source.suffix in {".sh", ".ps1", ".cmd"}:
                shutil.copy2(source, scripts / source.name)
        (project / "pyproject.toml").write_text(
            '[project]\nname="launcher-smoke"\nversion="0.0.0"\n'
            'requires-python=">=3.13,<3.14"\ndependencies=[]\n'
            '[tool.uv]\npackage=false\n'
        )
        # This stand-in captures the launcher's actual interpreter, cwd and argument forwarding.
        (scripts / "manage.py").write_text(
            'import json,os,sys\nfrom pathlib import Path\n'
            'Path("capture.json").write_text(json.dumps({"args":sys.argv[1:],'
            '"cwd":os.getcwd(),"prefix":sys.prefix,"version":list(sys.version_info[:2]),'
            '"utf8":os.getenv("PYTHONUTF8")}))\n'
            'sys.exit(7 if "--fail" in sys.argv else 0)\n'
        )
        subprocess.run([uv, "lock", "--project", str(project)], check=True, capture_output=True)
        env = os.environ.copy()
        other_environment = temp / "must not be used"
        env["UV_PROJECT_ENVIRONMENT"] = str(other_environment)
        env["VIRTUAL_ENV"] = str(temp / "another active environment")
        shells = [("sh", ["sh"])]
        if os.name == "nt":
            shells = [("cmd", ["cmd.exe", "/d", "/c"])]
            ps = shutil.which("pwsh") or shutil.which("powershell.exe")
            if ps:
                shells.append(("ps1", [ps, "-NoProfile", "-File"]))
        for suffix, prefix in shells:
            for command in ["setup", "start", "test"]:
                extra = ["--solution", "--port", "27861", "argument with spaces"]
                result = subprocess.run(prefix + [str(scripts / f"{command}.{suffix}"), *extra],
                                        cwd=temp, env=env, capture_output=True, text=True)
                assert result.returncode == 0, result.stdout + result.stderr
                capture = json.loads((project / "capture.json").read_text())
                assert capture["args"] == [command, *extra], capture
                assert Path(capture["cwd"]).resolve() == project.resolve(), capture
                assert Path(capture["prefix"]).resolve() == (project / ".venv").resolve(), capture
                assert capture["version"] == [3, 13], capture
                assert capture["utf8"] == "1", capture
                assert not other_environment.exists()
            result = subprocess.run(prefix + [str(scripts / f"start.{suffix}"), "--fail"],
                                    cwd=temp, env=env, capture_output=True, text=True)
            assert result.returncode == 7, (result.returncode, result.stdout, result.stderr)
            print(f"PASS: {suffix} launchers: spaces, cwd, arguments, isolated Python 3.13 and exit status.")


if __name__ == "__main__":
    main()
