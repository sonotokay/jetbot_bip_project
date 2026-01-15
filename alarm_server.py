#!/usr/bin/env python3
import os
import time
import subprocess
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# TODO Wave sound 
ALARM_WAV = os.environ.get("ALARM_WAV", "/home/jetbot/alarm.wav")

# Alarm cooldown in seconds
COOLDOWN_SEC = float(os.environ.get("ALARM_COOLDOWN", "2.0"))
_last_alarm_ts = 0.0
_lock = Lock()


class AlarmMsg(BaseModel):
    alarm: bool = True
    reason: str | None = None
    source: str | None = None


def play_alarm():
    """Spielt Alarmton ab (nicht-blockierend)."""
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
