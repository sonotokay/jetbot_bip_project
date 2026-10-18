#!/usr/bin/env python3
import argparse
import subprocess
import re
import signal

PID = None  # Process ID of the jetbot driver (set via --pid)
FD = "13"   # File descriptor of /dev/ttyACM0 (set via --fd)

def spy():
    print(f"[*] Attaching to Process {PID} to spy on File Descriptor {FD}...")
    print("[*] Note: This uses 'sudo' internally. Enter password if asked.")
    print("[*] Press Ctrl+C to stop.")
    print("-" * 50)

    # Command breakdown:
    # -f            : Follow all threads (needed for multi-threaded Python processes)
    # -p PID        : Attach to the process
    # -e trace=read : Only trace 'read' operations
    # -s 200        : Capture up to 200 characters of data
    cmd = ["sudo", "strace", "-f", "-p", str(PID), "-e", "trace=read", "-s", "200"]

    try:
        # Open strace and capture standard error (where strace prints output)
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)

        while True:
            line = process.stderr.readline()
            if not line:
                break # Process died
            
            # We are looking for: read(13, "some_data", ...)
            # The regex matches: read( followed by our FD, followed by comma
            if f"read({FD}," in line:
                # Extract the text inside the quotes
                match = re.search(r'\"(.*?)\"', line)
                if match:
                    data = match.group(1)
                    
                    # Clean up common escape codes
                    clean_data = data.replace("\\r", "").replace("\\n", "").replace("\\x", "")
                    
                    # Look for voltage patterns (e.g. 11.1, 12.5)
                    # We accept numbers between 8.0 and 13.0
                    volts = re.findall(r"([0-9]+\.[0-9]+)", clean_data)
                    for v in volts:
                        val = float(v)
                        if 8.0 < val < 13.0:
                            percent = max(0, min(100, (val - 10.5) / (12.6 - 10.5) * 100))
                            
                            # Print result clearly
                            print(f"\n>>> BATTERY DATA FOUND: {val}V ({percent:.1f}%)")
                            print(f"    Raw Line: {clean_data}")
                            
                            # Cleanup and Exit after first success
                            process.send_signal(signal.SIGINT)
                            return

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read JetBot battery voltage via strace")
    parser.add_argument('--pid', type=int, required=True, help="PID of the jetbot driver process")
    parser.add_argument('--fd', default="13", help="File descriptor of /dev/ttyACM0 (default: 13)")
    args = parser.parse_args()
    
    PID, FD = args.pid, args.fd
    spy()