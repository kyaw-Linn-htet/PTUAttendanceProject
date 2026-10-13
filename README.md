# PTU Face Recognition — Deployment Notes

## Quick start
- **Local:** `pip install -r requirements.txt && streamlit run streamlit_app.py`
- **Streamlit Cloud:** push this folder to a repo and deploy `streamlit_app.py`.
  `requirements.txt` and `packages.txt` are auto-detected by filename —
  no extra configuration needed.

## Two face-detection engines (auto-detected, no code changes needed)
| Engine | Requires | Accuracy | Install reliability |
|---|---|---|---|
| **OpenCV (Haar cascade)** — default | `opencv-python-headless` only | Good | Installs from a wheel in seconds, always works |
| **face_recognition (dlib)** — optional | `dlib` (compiles from source) | Best | Can take 10-20+ min to build; can fail on constrained hosts |

`requirements.txt` ships with the OpenCV-only path so the app is
**guaranteed to deploy successfully**. If you want the higher-accuracy
dlib engine and can tolerate a slower/riskier build, swap in
`requirements-full.txt` (see comments inside that file). The Python
code in `streamlit_app.py` auto-detects whichever engine(s) are
actually importable at runtime — nothing else needs to change.

## Root causes of the original bugs (see full explanation in chat)
1. **`No module named 'cv2'`** — the requirements file was named
   `requirements_streamlit.txt`. Streamlit Cloud only auto-installs
   from a file named exactly `requirements.txt`, so none of the
   dependencies (including `opencv-python-headless`) were ever
   installed.
2. A second, independent bug: unpinned `opencv-python-headless`
   resolves to `5.0.0`, which **removes/breaks
   `cv2.CascadeClassifier`** — fixed with a `<5.0.0` pin.
3. If `dlib` had been included and its source build failed for any
   reason, pip aborts the *entire* `requirements.txt` install —
   verified empirically — which would silently take `cv2` down with
   it. Fixed by keeping the default `requirements.txt` free of any
   package that requires compiling from source.
4. **"0 valid → skipped" on dependency errors** — the old
   `encode_face()` caught a missing-`cv2` `ImportError` inside the
   "no face detected" fallback path, so a dependency outage was
   mislabeled as "no face in the photo". Now `ERR_ENGINE_UNAVAILABLE`
   is a distinct, clearly-messaged error code, separate from
   `ERR_NO_FACE`.
5. **Sidebar contrast** — "USER PANEL"/"ADMIN PANEL" labels were
   `opacity:.45`; sidebar buttons relied on a blanket
   `color:#fff!important` with no explicit background, so default
   button chrome fought the dark theme. Both now have explicit,
   accessible colors for normal/hover/active states.

## Local Flask classroom attendance

Install dependencies in an activated virtual environment:

```powershell
python -m pip install -r requirements.txt
```

Run Flask on the computer connected to the webcam:

```powershell
$env:FLASK_SECRET_KEY = 'replace-with-a-long-random-secret'
$env:ADMIN_USERNAME = 'PTUG4'
$env:ADMIN_PASSWORD = 'change-this-password'
$env:CAMERA_INDEX = '0'
python app.py
```

Open `http://127.0.0.1:5000/admin/login`, sign in, then choose **Attendance**.
For other devices on a trusted local network, use `http://<computer-LAN-IP>:5000/admin/login`; the app binds to `0.0.0.0`. The webcam is attached to the Flask computer, not to each browser. Limit firewall access to a trusted private network and do not expose Flask's development server publicly.

### Optional Google Sheets export

1. In Google Cloud Console, enable the Sheets API, create a service account and download its JSON key to a private path. Share the destination spreadsheet with the service account email as an editor. Keep the key outside source control.
2. Set these variables before starting Flask:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\private\service-account.json'
$env:GOOGLE_SHEETS_SPREADSHEET_ID = 'spreadsheet-id-from-its-url'
$env:GOOGLE_SHEETS_WORKSHEET = 'Attendance'
```

3. End a session and click **Export to Google Sheets**. Repeated exports skip local attendance IDs already in column A. Failed exports preserve local data and can be retried.

Sessions snapshot students matching the selected **major and semester**, because the existing student schema has no class enrollment table. The confidence formula is retained and its raw value is checked against 50 and 60 before display rounding. The old maximum-distance filter prevented confidence above 40%; best-match selection now happens before applying the existing confidence bands.

### Attendance implementation checklist

- [ ] Start with camera 0, then try another configured index; confirm an unavailable camera reports an error and the Stop button releases it.
- [ ] Check face feedback and confidence display. Confirm under 50 shows exactly `The Student has not been recorded in the Database`; 50 through below 60 shows exactly `Please Make More Clear Photo`; 60 or higher shows a match.
- [ ] Keep one student visible across multiple frames and confirm one record per session. Restart a session and confirm it can produce a separate record.
- [ ] Stop and restart Flask; confirm SQLite attendance remains.
- [ ] End a session and verify absent students come from the captured major/semester roster; verify attendance percentage and teacher correction.
- [ ] Configure Sheets credentials and destination, export twice, and confirm the second export adds no duplicate rows. Disconnect network or use invalid credentials and confirm the local record remains available for retry.
