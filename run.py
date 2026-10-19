"""Start the unified backend, or the legacy three-port setup on request.

Run ``python run.py`` for one FastAPI listener. Use
``python run.py --legacy-three-services`` only for the old local setup.
Existing application data and databases are never changed by this launcher.
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, threading, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SERVICES = (("task1", "task1_knowledge_brain.main:app", 8000), ("task2", "task2_inbox_agent.main:app", 8001), ("task3", "task3_meeting_scope.main:app", 8002))

def available(port):
    with socket.socket() as sock:
        try: sock.bind(("127.0.0.1", port)); return True
        except OSError: return False

def stream(name, process):
    if process.stdout:
        for line in process.stdout: print(f"[{name}] {line.rstrip()}", flush=True)

def healthy(port, unified=False):
    try:
        path = "/api/v1/unified-health" if unified else "/api/v1/health"
        with urlopen(Request(f"http://127.0.0.1:{port}{path}"), timeout=2) as response:
            data=json.loads(response.read().decode("utf-8")); return response.status == 200 and data.get("status") == "ok"
    except (OSError, URLError, HTTPError, ValueError): return False

def stop(processes):
    live=[p for p in processes.values() if p.poll() is None]
    for process in live: process.terminate()
    for process in live:
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--health-timeout", type=float, default=60)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--legacy-three-services", action="store_true", help="Use the old three-port development setup")
    args=parser.parse_args()
    services = SERVICES if args.legacy_three_services else (("unified", "unified_backend:app", args.port),)
    blocked=[str(port) for _,_,port in services if not available(port)]
    if blocked: print("Cannot start; port(s) already in use: "+", ".join(blocked), file=sys.stderr); return 2
    processes={}
    try:
        for name,target,port in services:
            child_env = os.environ.copy()
            if name == "unified":
                # Keep Task 2/3 HTTP clients on this exact listener even when
                # a stale TASK1_API_BASE from the legacy setup is present.
                child_env["PORT"] = str(port)
            process=subprocess.Popen([sys.executable,"-m","uvicorn",target,"--host","127.0.0.1","--port",str(port)],cwd=ROOT,env=child_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
            processes[name]=process; threading.Thread(target=stream,args=(name,process),daemon=True).start(); print(f"Started {name} on port {port} (pid {process.pid})",flush=True)
        deadline=time.monotonic()+args.health_timeout
        for name,_,port in services:
            is_unified = name == "unified"
            while time.monotonic()<deadline and processes[name].poll() is None and not healthy(port, is_unified): time.sleep(.5)
            if not healthy(port, is_unified): raise RuntimeError(f"{name} did not become healthy")
            health_path = "/api/v1/unified-health" if is_unified else "/api/v1/health"
            print(f"{name} healthy: http://127.0.0.1:{port}{health_path}",flush=True)
        print("All services are ready. Press Ctrl+C to stop them.",flush=True)
        while True:
            for name,process in processes.items():
                if process.poll() is not None: raise RuntimeError(f"{name} stopped unexpectedly (code {process.returncode})")
            time.sleep(1)
    except KeyboardInterrupt: print("Shutdown requested.",flush=True); return 0
    except RuntimeError as exc: print(f"Startup failed: {exc}",file=sys.stderr); return 1
    finally: stop(processes)

if __name__ == "__main__": raise SystemExit(main())
