import json
import os
import shutil
import subprocess
import time
from pathlib import Path

KERNEL_ID = "mishajaiswal0511/ecovision-yolo11m-multi-object-training"

def monitor():
    workspace = Path(__file__).resolve().parent.parent
    cli_path = workspace / ".venv" / "Scripts" / "kaggle.exe"
    log_file = workspace / "training" / "kaggle" / "training_execution.log"
    out_dir = workspace / "models" / "experiments" / "yolo11m_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  KAGGLE TRAINING MONITOR — BACKGROUND RUNNER (UTF-8)")
    print(f"  Kernel: {KERNEL_ID}")
    print("=" * 70)

    start_time = time.time()
    poll_interval = 60  # seconds

    utf8_env = os.environ.copy()
    utf8_env["PYTHONIOENCODING"] = "utf-8"

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n--- MONITOR STARTED at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    while True:
        try:
            status_res = subprocess.run(
                [str(cli_path), "kernels", "status", KERNEL_ID],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=utf8_env
            )
            status_text = status_res.stdout.strip()
            elapsed_min = (time.time() - start_time) / 60

            msg = f"[{time.strftime('%H:%M:%S')}] Elapsed: {elapsed_min:.1f}m | Status: {status_text}"
            print(msg, flush=True)
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
                lf.flush()

            # Check if finished
            if "COMPLETE" in status_text.upper():
                print("\n[SUCCESS] Training Kernel completed successfully!")
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"\n--- TRAINING COMPLETED at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

                # Retrieve outputs
                print(f"Retrieving outputs to {out_dir}...")
                dl_res = subprocess.run(
                    [str(cli_path), "kernels", "output", KERNEL_ID, "-p", str(out_dir)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=utf8_env
                )
                print(f"Download stdout: {dl_res.stdout}")
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"Download stdout: {dl_res.stdout}\n")

                # Verify yolo11m_best.pt
                downloaded_best = out_dir / "yolo11m_best.pt"
                candidate_root_best = workspace / "models" / "experiments" / "yolo11m_best.pt"
                if downloaded_best.exists():
                    shutil.copy2(downloaded_best, candidate_root_best)
                    print(f"Candidate best checkpoint copied to: {candidate_root_best}")

                # Save completion report
                total_duration_hours = (time.time() - start_time) / 3600
                report = {
                    "kernel_id": KERNEL_ID,
                    "status": "COMPLETED",
                    "duration_hours": round(total_duration_hours, 2),
                    "output_dir": str(out_dir),
                    "candidate_checkpoint": str(candidate_root_best),
                    "checkpoint_exists": candidate_root_best.exists(),
                    "checkpoint_size_mb": round(candidate_root_best.stat().st_size / (1024**2), 2) if candidate_root_best.exists() else 0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(workspace / "docs" / "phase3_training_completion_report.json", "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)

                print("Phase 3 Execution & Artifact Retrieval Complete!")
                return 0

            elif "ERROR" in status_text.upper() or "CANCEL" in status_text.upper():
                err_msg = f"[FAILURE] Kernel terminated with status: {status_text}"
                print(err_msg)
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(err_msg + "\n")

                # Fetch logs
                logs_res = subprocess.run(
                    [str(cli_path), "kernels", "logs", KERNEL_ID],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=utf8_env
                )
                print("--- KERNEL ERROR LOGS ---")
                print(logs_res.stdout[-2000:] if logs_res.stdout else "No stdout logs")
                print(logs_res.stderr[-2000:] if logs_res.stderr else "No stderr logs")
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write("--- ERROR LOGS ---\n" + (logs_res.stdout or "") + "\n" + (logs_res.stderr or "") + "\n")

                return 1

        except Exception as e:
            print(f"Monitor exception: {e}")
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"Monitor exception: {e}\n")

        time.sleep(poll_interval)

if __name__ == "__main__":
    import sys
    sys.exit(monitor())
