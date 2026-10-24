# sleep-detect-alarm
Detects when you fall asleep at your desk using your webcam. Shows a green face box when you're awake and a red pulsing alert when your eyes are closed too long — and plays an alarm sound to wake you up.

---

## Requirements

- Python 3.8+
- Webcam

Install dependencies:

```bash
pip install mediapipe opencv-python numpy pygame
```

---

## Setup

1. **Clone / download** this folder.
2. **Add your alarm sound** — place any `.mp3` or `.wav` file in the project folder and update the path in `sleep_alarm.py`:

```python
ALARM_SOUND_FILE = r"/full/path/to/your/alarm.mp3"
```

3. **(Optional) Tune the settings** at the top of `sleep_alarm.py`:

| Variable | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.22` | Eye openness threshold — lower = less sensitive |
| `EYE_CLOSED_SECONDS` | `2.5` | Seconds eyes must stay closed to trigger alarm |

---

## Run

```bash
python3 sleep_alarm.py
```

Press **Q** to quit.

---

## How It Works

| Status | Signal | What's shown |
|---|---|---|
| 👁️ Eyes open | 🟢 Green | Green face box + "AWAKE" pill |
| 😴 Eyes closing | 🟠 Orange bar | Progress bar filling up |
| 🚨 Eyes closed ≥ 2.5s | 🔴 Red | Pulsing red overlay + alarm sound |

- Uses **MediaPipe Face Mesh** to track 468 facial landmarks in real time.
- Computes the **Eye Aspect Ratio (EAR)** for both eyes — when EAR drops below the threshold, the countdown starts.
- Alarm loops until you open your eyes.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Black screen / no camera | Change `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` in `sleep_alarm.py` |
| Alarm not playing | Check the file path in `ALARM_SOUND_FILE` is correct |
| Too many false triggers | Increase `EAR_THRESHOLD` to `0.25` |
| Alarm fires too quickly | Increase `EYE_CLOSED_SECONDS` to `3.0` or higher |
