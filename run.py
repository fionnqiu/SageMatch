"""Restart the SageMatch frontend and backend.

If a service is already running, stop that process tree and start it again.
If it is not running, start it directly. When the preferred port is taken by
something else, move to the next free port.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
RUNTIME_DIR = ROOT / ".runtime"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173


def python_executable() -> Path:
    """Prefer the project virtualenv so restarts do not depend on PATH."""
    candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def npm_executable() -> str:
    """Windows 不能直接创建 npm，因为它是 cmd 脚本。"""
    return "npm.cmd" if os.name == "nt" else "npm"


def listener_pids(port: int) -> list[int]:
    """Return process ids listening on this port. An empty list means it is free."""
    # netstat is available on Windows without extra tools and shows the owning pid.
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    pids: list[int] = []
    needle = f":{port}"
    for line in result.stdout.splitlines():
        if "LISTENING" not in line or needle not in line:
            continue
        parts = line.split()
        local = parts[1] if len(parts) > 1 else ""
        if not local.endswith(needle):
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


def command_line(pid: int) -> str:
    """Read one process command line. A dead process returns an empty string."""
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\").CommandLine",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip()


def is_our_service(pid: int, marker: str) -> bool:
    """Only treat a listener as ours when its command belongs to this checkout."""
    command = command_line(pid)
    return marker in command and str(ROOT).lower() in command.lower()


def stop_pid(pid: int) -> None:
    """Stop a process and the children it spawned, such as uvicorn --reload."""
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def port_is_free(port: int) -> bool:
    """Bind the port briefly. Listening pids can lag behind a process that just exited."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def choose_port(preferred: int, marker: str) -> int:
    """Reuse the preferred port after stopping our own process. Otherwise step forward."""
    port = preferred
    while port < preferred + 20:
        pids = listener_pids(port)
        if not pids and port_is_free(port):
            return port
        if pids and all(is_our_service(pid, marker) for pid in pids):
            for pid in pids:
                stop_pid(pid)
            for _ in range(20):
                if port_is_free(port):
                    return port
                time.sleep(0.25)
        port += 1
    raise RuntimeError(f"从 {preferred} 起连续 20 个端口都不可用")


def wait_until_ready(url: str, timeout: float = 25) -> bool:
    """A restart is only useful once the service actually answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.4)
    return False


def _spawn(name: str, args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Start one service in the background and keep its output for the next restart."""
    log = open(RUNTIME_DIR / f"{name}.log", "w", encoding="utf-8")
    return subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def start_backend(port: int, frontend_port: int) -> subprocess.Popen[bytes]:
    """Launch uvicorn from backend/ so app.main imports without extra PYTHONPATH."""
    # 前端端口要先定下来，CORS 才能放行实际打开页面的地址。
    env = os.environ.copy()
    env["SAGEMATCH_CORS_ORIGIN"] = f"http://127.0.0.1:{frontend_port},http://localhost:{frontend_port}"
    return _spawn(
        "backend",
        [
            str(python_executable()),
            "-m",
            "uvicorn",
            "app.main:app",
            "--reload",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        BACKEND_DIR,
        env,
    )


def start_frontend(port: int, api_port: int) -> subprocess.Popen[bytes]:
    """Point the dev proxy at the backend port chosen in this same run."""
    env = os.environ.copy()
    env["SAGEMATCH_API_PORT"] = str(api_port)
    return _spawn(
        "frontend",
        [npm_executable(), "run", "dev", "--", "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
        FRONTEND_DIR,
        env,
    )


def main() -> int:
    RUNTIME_DIR.mkdir(exist_ok=True)
    backend_running = bool(listener_pids(BACKEND_PORT))
    frontend_running = bool(listener_pids(FRONTEND_PORT))
    print(f"后端 {'已在运行，先关闭再启动' if backend_running else '未运行，直接启动'}")
    print(f"前端 {'已在运行，先关闭再启动' if frontend_running else '未运行，直接启动'}")

    # 两个端口都先选好。先停占用者再启动，这样 CORS 和前端代理用的是同一对端口。
    backend_port = choose_port(BACKEND_PORT, "uvicorn")
    frontend_port = choose_port(FRONTEND_PORT, "vite")
    backend = start_backend(backend_port, frontend_port)
    backend_ready = wait_until_ready(f"http://127.0.0.1:{backend_port}/docs")

    frontend = start_frontend(frontend_port, backend_port)
    frontend_ready = wait_until_ready(f"http://127.0.0.1:{frontend_port}/")

    (RUNTIME_DIR / "services.env").write_text(
        f"BACKEND_PORT={backend_port}\nFRONTEND_PORT={frontend_port}\n",
        encoding="utf-8",
    )
    print(f"后端 http://127.0.0.1:{backend_port}/docs  {'就绪' if backend_ready else '尚未响应'}")
    print(f"前端 http://127.0.0.1:{frontend_port}/  {'就绪' if frontend_ready else '尚未响应'}")
    if backend.poll() is not None or frontend.poll() is not None or not backend_ready or not frontend_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
