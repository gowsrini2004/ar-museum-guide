import subprocess
import time
import sys
import os
import threading

def stream_logs(pipe, prefix, color_code):
    """Function to read from a pipe and print with a prefix in a specific color."""
    reset_color = "\033[0m"
    try:
        for line in iter(pipe.readline, ''):
            if line:
                print(f"{color_code}[{prefix}]{reset_color} {line.strip()}")
    except Exception as e:
        print(f"Error streaming logs for {prefix}: {e}")

def start_services():
    print("============================================================")
    print("           AR Museum Guide - Unified Log Center             ")
    print("============================================================\n")

    # ANSI Color codes for prettier logs
    COLORS = {
        "ML": "\033[94m",     # Blue
        "TRAIN": "\033[92m",  # Green
        "QA": "\033[93m",     # Yellow
        "FRONT": "\033[95m",  # Magenta
        "ERR": "\033[91m"     # Red
    }

    services = [
        {"name": "ML", "cmd": [sys.executable, "-m", "uvicorn", "backend.ml_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"], "color": COLORS["ML"]},
        {"name": "TRAIN", "cmd": [sys.executable, "-m", "uvicorn", "backend.training_api:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"], "color": COLORS["TRAIN"]},
        {"name": "QA", "cmd": [sys.executable, "-m", "uvicorn", "backend.qa_api:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info"], "color": COLORS["QA"]},
        {"name": "FRONT", "cmd": [sys.executable, "run_ar_server.py"], "color": COLORS["FRONT"]}
    ]

    processes = []
    threads = []

    try:
        for service in services:
            print(f"[*] Starting {service['name']} Service...")
            
            # Start process with stdout/stderr redirected to pipe
            p = subprocess.Popen(
                service['cmd'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )
            processes.append({"name": service['name'], "process": p})
            
            # Start a thread to stream logs for this service
            t = threading.Thread(target=stream_logs, args=(p.stdout, service['name'], service['color']), daemon=True)
            t.start()
            threads.append(t)
            
            time.sleep(1.5) # Stagger start slightly

        print("\n" + "="*60)
        print(" [READY] All services active. Logs streaming below:")
        print("="*60 + "\n")

        # Keep main thread alive and monitor for crashes
        while True:
            for p_info in processes:
                p = p_info['process']
                if p.poll() is not None:
                    print(f"\n\033[91m[CRASH] {p_info['name']} has stopped unexpectedly with exit code {p.returncode}\033[0m")
                    # Optionally we could restart here, but for now we just notify
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n[*] Shutting down all services...")
        for p_info in processes:
            p_info['process'].terminate()
        print("[OK] Stopped.")

if __name__ == "__main__":
    start_services()
