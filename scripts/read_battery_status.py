#!/usr/bin/env python3
import subprocess
import re
import sys
import os
import signal

# --- HARDCODED CONFIG (Based on your screenshots) ---
PID = 10627          # The process ID of 'jetbot'
FD  = "13"           # The File Descriptor for /dev/ttyACM0
# ----------------------------------------------------

def spy():
    print(f"[*] Attaching to Process {PID} to spy on File Descriptor {FD}...")
    print("[*] Note: This uses 'sudo' internally. Enter password if asked.")
    print("[*] Press Ctrl+C to stop.")
    print("-" * 50)

    # Command breakdown:
    # -f            : Follow all threads (CRITICAL for Python apps)
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
            
            # DEBUG: Uncomment the next line if you still see nothing
            # print(f"DEBUG: {line.strip()}")

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
    spy()