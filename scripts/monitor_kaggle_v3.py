import json
import os
import shutil
import subprocess
import sys
import time
import queue
import threading
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

KERNEL_ID = "mishajaiswal0511/ecovision-yolo11m-multi-object-training"

def main():
    workspace = Path(__file__).resolve().parent.parent
    cli_path = workspace / ".venv" / "Scripts" / "kaggle.exe"
    log_file = workspace / "training" / "kaggle" / "training_execution.log"
    stream_dump_file = workspace / "training" / "kaggle" / "live_stream_v3.txt"
    out_dir = workspace / "models" / "experiments" / "yolo11m_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("=" * 70)
    print("  KAGGLE TRAINING LIVE MONITOR V3")
    print(f"  Kernel: {KERNEL_ID}")
    print("=" * 70)

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n--- MONITOR V3 STARTED at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    seen_events = 0
    start_time = time.time()
    
    while True:
        try:
            # 1. Check status
            status_obj = api.kernels_status(KERNEL_ID)
            status_text = str(status_obj.status) if hasattr(status_obj, 'status') else str(status_obj)
            elapsed_min = (time.time() - start_time) / 60
            
            # 2. Poll new SSE events with 4-second timeout
            q = queue.Queue()
            def stream_worker():
                try:
                    for ev in api.kernels_logs_stream(KERNEL_ID):
                        q.put(ev)
                except Exception as ex:
                    q.put(ex)

            t = threading.Thread(target=stream_worker, daemon=True)
            t.start()
            
            new_events = []
            while True:
                try:
                    item = q.get(timeout=3.0)
                    if isinstance(item, Exception):
                        break
                    new_events.append(item)
                except queue.Empty:
                    break
                    
            if len(new_events) > seen_events:
                added = new_events[seen_events:]
                with open(stream_dump_file, "a", encoding="utf-8") as sdf, open(log_file, "a", encoding="utf-8") as lf:
                    for ev in added:
                        t_sec = ev.get("time", 0)
                        sname = ev.get("stream_name", "")
                        data = ev.get("data", "")
                        line = f"[{t_sec:6.1f}s] [{sname}] {data}"
                        sdf.write(line)
                        lf.write(line)
                        # Print relevant lines
                        clean = data.strip()
                        if any(k in clean for k in ['GPU', 'Tesla', 'Dataset staging', 'Split', 'Writable', 'Configured runtime', 'Epoch', 'epoch', 'loss', 'mAP', 'images', 'labels']):
                            print(f"[{t_sec:6.1f}s] {clean}")
                seen_events = len(new_events)

            msg = f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed_min:.1f}m | Status: {status_text} | Total Events: {seen_events}"
            print(msg, flush=True)

            if "COMPLETE" in status_text.upper():
                print("\n[SUCCESS] Kernel completed successfully!")
                dl_res = subprocess.run(
                    [str(cli_path), "kernels", "output", KERNEL_ID, "-p", str(out_dir)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )
                print(f"Download output: {dl_res.stdout}")
                
                downloaded_best = out_dir / "yolo11m_best.pt"
                candidate_root_best = workspace / "models" / "experiments" / "yolo11m_best.pt"
                if downloaded_best.exists():
                    shutil.copy2(downloaded_best, candidate_root_best)
                    print(f"Candidate best checkpoint copied to: {candidate_root_best}")
                    
                return 0

            elif "ERROR" in status_text.upper() or "CANCEL" in status_text.upper():
                print(f"\n[FAILURE] Kernel terminated with status: {status_text}")
                return 1

        except Exception as e:
            print(f"Monitor error: {e}", flush=True)

        time.sleep(15)

if __name__ == "__main__":
    sys.exit(main())

