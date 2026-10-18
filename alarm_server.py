#!/usr/bin/env python3
import os
import time
import subprocess
from threading import Lock
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Alarm sound, defaults to the one shipped in ressources/
DEFAULT_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ressources", "alarm.wav")
ALARM_WAV = os.environ.get("ALARM_WAV", DEFAULT_WAV)

# Alarm cooldown in seconds
COOLDOWN_SEC = float(os.environ.get("ALARM_COOLDOWN", "2.0"))
_last_alarm_ts = 0.0
_lock = Lock()


class AlarmMsg(BaseModel):
    alarm: bool = True
    reason: Optional[str] = None
    source: Optional[str] = None


def play_alarm():
    """Play the alarm sound (non-blocking)."""
    if not os.path.exists(ALARM_WAV):
        subprocess.Popen(["bash", "-lc", "printf '\\a'"])
        return

    subprocess.Popen(["aplay", "-q", ALARM_WAV])


@app.post("/alarm")
def alarm(msg: AlarmMsg):
    global _last_alarm_ts

    if not msg.alarm:
        return {"ok": True, "played": False, "note": "alarm=false"}

    with _lock:
        now = time.time()
        if now - _last_alarm_ts < COOLDOWN_SEC:
            return {"ok": True, "played": False, "note": "cooldown"}
        _last_alarm_ts = now

    play_alarm()
    return {"ok": True, "played": True, "reason": msg.reason, "source": msg.source}


@app.get("/health")
def health():
    return {"ok": True, "alarm_wav": ALARM_WAV, "cooldown_sec": COOLDOWN_SEC}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)