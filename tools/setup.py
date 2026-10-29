"""Create a local environment and retrieve exactly the upstream revision used by the demo."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = "ca13310e2f9f788d1dd3feeea8213c1b0c23913c"


def run(*command):
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    upstream = ROOT / ".upstream/qwenputer"
    if not upstream.exists():
        upstream.parent.mkdir(exist_ok=True)
        run("git", "clone", "https://github.com/divagr18/qwenputer.git", str(upstream))
    current = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True)
    if changes:
        raise SystemExit("Upstream checkout has local changes. Preserve them before setup.")
    if current != REVISION:
        run("git", "-C", str(upstream), "fetch", "origin", REVISION)
        run("git", "-C", str(upstream), "checkout", "--detach", REVISION)
    venv = ROOT / ".venv"
    if not venv.exists():
        run(sys.executable, "-m", "venv", str(venv))
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run(str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"))
    run(str(python), "-m", "pip", "install", "--no-deps", "-e", str(upstream))
    print("Ready. Follow REPRODUCE.md, starting with experiments/memory-map.")


if __name__ == "__main__":
    main()
