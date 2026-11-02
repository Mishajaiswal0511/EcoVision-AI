import os
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def main():
    api = KaggleApi()
    api.authenticate()
    kernel_id = 'mishajaiswal0511/ecovision-yolo11m-multi-object-training'
    
    print(f"Connecting to live logs stream for: {kernel_id}")
    dump_path = 'training/kaggle/live_stream_dump.txt'
    
    count = 0
    with open(dump_path, 'w', encoding='utf-8', errors='replace') as f:
        try:
            stream = api.kernels_logs_stream(kernel_id)
            for ev in stream:
                count += 1
                t = ev.get('time', 0)
                name = ev.get('stream_name', '')
                data = ev.get('data', '')
                line = f"[{t:.2f}s] [{name}] {data}"
                f.write(line)
                f.flush()
        except Exception as e:
            print(f"Stream error: {e}")
            
    print(f"Total events collected: {count}")
    
    # Read back and print the last 60 lines from dump_path
    with open(dump_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        print(f"Total lines in dump: {len(lines)}")
        print("\n=== LAST 60 LINES OF STREAM DUMP ===")
        for l in lines[-60:]:
            print(l, end='')

if __name__ == '__main__':
    main()
