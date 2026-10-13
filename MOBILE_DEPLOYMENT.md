# Streamlit mobile deployment and testing

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub, including the root `requirements.txt` and `packages.txt`.
2. In Streamlit Community Cloud, create an app from the repository and select its branch.
3. Set **Main file path** to `streamlit_app.py` (repository root), then deploy.
4. If using Google Sheets export, add credentials through Cloud secrets/settings; do not commit service-account keys.
5. SQLite data on Community Cloud can be lost when the app restarts or is rebuilt. Use persistent storage for production attendance records.

The app starts with its navigation sidebar collapsed to leave more room on phones. Open it with Streamlit's menu button. `st.camera_input` requests access to the browser user's camera and returns a still image; it does not connect to a webcam attached to the Cloud server. On mobile, allow camera permission when prompted. If that control is unavailable, use the upload tab and choose the phone camera in the system image picker. Recognition runs on the server after upload.

## Manual mobile checklist

- Check pages at 320 px, 375 px, and 430 px widths, then at tablet and desktop widths. Confirm there is no page-level horizontal scrolling.
- Open and close navigation; visit Home, Search, Face Detection, Admin, Students, and Attendance.
- On Android Chrome and iPhone Safari over HTTPS, grant camera permission and capture a photo. Also deny permission and verify upload remains usable.
- Upload a phone photo with rotation metadata and confirm the preview has the correct orientation and aspect ratio.
- Submit search, registration, login, and attendance forms with valid and missing values; check that controls are comfortably tappable.
- Check confidence below 50%, from 50% inclusive to below 60%, and at 60% or higher. Low-confidence results must not create attendance; comparisons use the unrounded confidence value.
- Capture the same attendance photo again and confirm one check-in per student/session. Review present records, the ended-session absent list, and the summary cards.
- Review student and attendance records on a narrow screen; use search and expandable records without needing a wide table.

These are manual device checks and have not been run as part of the code change.
