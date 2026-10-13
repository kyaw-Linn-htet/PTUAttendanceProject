"""
PTU Student Face Recognition System — Streamlit Version
Fixed: use_container_width, image handling, camera capture,
       face encoding consistency, training pipeline, error handling.
"""

import streamlit as st
import sqlite3
import os
import pickle
import base64
import io
import csv
import logging
import hashlib
import re
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ── Logging (visible in Streamlit Cloud logs) ────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("PTU")

try:
    import cv2  # noqa: F401
    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = None
    CV2_VERSION = cv2.__version__
except Exception as _cv2_exc:                       
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(_cv2_exc)
    CV2_VERSION = None
    log.warning("cv2 (OpenCV) is not available: %s", CV2_IMPORT_ERROR)

try:
    import face_recognition  
    FACE_RECOGNITION_AVAILABLE = True
    FACE_RECOGNITION_IMPORT_ERROR = None
except Exception as _fr_exc:                         
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_IMPORT_ERROR = str(_fr_exc)
    log.warning("face_recognition is not available: %s",
                FACE_RECOGNITION_IMPORT_ERROR)

FACE_ENGINE_AVAILABLE = CV2_AVAILABLE or FACE_RECOGNITION_AVAILABLE


def face_engine_status() -> dict:
    """Small diagnostics dict used by the UI / troubleshooting panel."""
    return {
        "cv2_available": CV2_AVAILABLE,
        "cv2_version": CV2_VERSION,
        "cv2_error": CV2_IMPORT_ERROR,
        "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
        "face_recognition_error": FACE_RECOGNITION_IMPORT_ERROR,
        "any_engine_available": FACE_ENGINE_AVAILABLE,
    }

st.set_page_config(
    page_title="PTU Face Recognition",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "PTUG4")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "PTU2026")
DATABASE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "ptu_students.db")
MYANMAR_TZ = timezone(timedelta(hours=6, minutes=30), name="MMT")
LOW_CONFIDENCE_THRESHOLD = 50.0
MEDIUM_CONFIDENCE_THRESHOLD = 60.0
ALLOWED_TYPES   = ["jpg", "jpeg", "png"]

MAJORS = [
    "Civil Engineering",
    "Electronic Engineering",
    "Mechanical Engineering",
    "Electrical Power Engineering",
    "Computer Engineering and Information Technology",
]
SEMESTERS = [
    "Semester","I","II","III","IV","V","VI",
    "VII","VIII","IX","X","XI","XII"
]

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f8f9fa}
[data-testid="stMain"] .block-container{width:100%;max-width:1200px;padding:1.5rem clamp(.75rem,3vw,2rem) 3rem}

/* Sidebar buttons: solid, readable in normal / hover / active states */
.ptu-card{background:#fff;border-radius:12px;padding:1.4rem;min-height:190px;height:100%;
          box-shadow:0 4px 12px rgba(0,0,0,.08);margin-bottom:1rem}
.ptu-card-blue {border-left:4px solid #0d6efd}
.ptu-card-green{border-left:4px solid #28a745}
.ptu-card-warn {border-left:4px solid #ffc107}
.hero{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
      border-radius:14px;padding:2.5rem 2rem;text-align:center;
      color:#fff;margin-bottom:1.5rem}
.hero h1{font-size:2rem;font-weight:700;margin:0}
.hero p{opacity:.8;margin:.5rem 0 0}
.result-card{background:#f0fff4;border:2px solid #28a745;
             border-radius:12px;padding:1.5rem}
.result-card h3{color:#1a7a4a;margin:0 0 .8rem}
.stat-card{border-radius:12px;padding:1.2rem 1.5rem;
           color:#fff;text-align:center}
.stat-num{font-size:2.2rem;font-weight:700}
.stat-lbl{font-size:.9rem;opacity:.85}
.info-row{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-top:1rem}
.info-box{background:#f4f6f7;border-radius:8px;padding:.7rem 1rem;
          border-left:3px solid #0d6efd}
.info-box small{color:#6c757d;font-size:.8rem}
.info-box b{display:block;font-size:1rem;color:#1a1a2e}
.badge-success{background:#28a745;color:#fff;padding:3px 10px;
               border-radius:20px;font-size:.8rem}
.badge-warn{background:#ffc107;color:#333;padding:3px 10px;
            border-radius:20px;font-size:.8rem}
/* Comfortable touch targets and fluid content at phone widths. */
button,[role="button"]{min-height:44px}
.bottom-nav-heading{font-size:.78rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#526178;padding:.55rem .25rem;border-top:1px solid #dbe2eb;margin-top:2rem}
[data-testid="stFileUploader"] section{min-height:72px}
[data-testid="stImage"] img{max-width:100%;height:auto;object-fit:contain}
.stDataFrame,[data-testid="stDataFrame"]{max-width:100%;overflow-x:auto}
@media(max-width:640px){
  [data-testid="stMain"] .block-container{padding:.75rem .75rem 2rem}
  h1{font-size:1.55rem!important} h2{font-size:1.35rem!important} h3{font-size:1.15rem!important}
  .hero{padding:1.25rem .85rem;border-radius:12px}
  .hero h1{font-size:1.45rem!important;line-height:1.25}
  .hero p{font-size:.9rem}
  .ptu-card{padding:1rem;margin-bottom:.65rem}
  .stat-card{padding:.85rem .65rem}
  .stat-num{font-size:1.7rem}
  .info-row{grid-template-columns:1fr;gap:.55rem}
  [data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:.6rem}
  [data-testid="stHorizontalBlock"]>[data-testid="column"]{flex:1 1 calc(50% - .6rem)!important;width:calc(50% - .6rem)!important;min-width:calc(50% - .6rem)!important}
  [data-testid="stCameraInput"] video,[data-testid="stCameraInput"] img{width:100%;height:auto;object-fit:contain}
  [data-testid="stDataFrame"]{font-size:.85rem}
}
@media(max-width:380px){
  [data-testid="stHorizontalBlock"]>[data-testid="column"]{flex:1 1 100%!important;width:100%!important;min-width:100%!important}
}
#MainMenu,footer,header{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# DATABASE  (mirrors database/db.py)
# ════════════════════════════════════════════════════════
def get_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT    NOT NULL,
            student_id     TEXT    UNIQUE NOT NULL,
            major          TEXT    NOT NULL,
            semester       TEXT    NOT NULL,
            roll_number    TEXT    NOT NULL,
            face_encodings BLOB,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_images (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image_data TEXT    NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            major TEXT NOT NULL,
            semester TEXT NOT NULL,
            session_date TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at TEXT,
            status TEXT NOT NULL DEFAULT 'active'
        );
        CREATE TABLE IF NOT EXISTS attendance_roster (
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            PRIMARY KEY (session_id, student_id),
            FOREIGN KEY (session_id) REFERENCES attendance_sessions(id),
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id INTEGER NOT NULL,
            session_date TEXT NOT NULL,
            check_in_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'present',
            recognition_confidence REAL NOT NULL,
            UNIQUE (student_id, session_id),
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (session_id) REFERENCES attendance_sessions(id)
        );
    """)
    conn.commit()
    return conn


def myanmar_now():
    """Return the current Myanmar Standard Time as a SQLite-friendly value."""
    return datetime.now(MYANMAR_TZ).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def myanmar_today():
    return datetime.now(MYANMAR_TZ).date().isoformat()


# ════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ════════════════════════════════════════════════════════
def bytes_to_rgb_array(img_bytes: bytes):
    """
    Safely convert raw bytes → PIL RGB Image → numpy uint8 array.
    Returns (array, None) on success, (None, error_str) on failure.
    """
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)  # fix phone-camera rotation
        pil_img = pil_img.convert("RGB")          # handles RGBA / L / P / CMYK
        arr = np.ascontiguousarray(
            np.array(pil_img, dtype=np.uint8))    # shape (H, W, 3)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None, "Image conversion failed: unexpected shape."
        return arr, None
    except UnidentifiedImageError:
        return None, "File is not a valid image."
    except Exception as exc:
        log.error("bytes_to_rgb_array: %s", exc)
        return None, f"Image error: {exc}"


def safe_read_uploaded(uploaded_file):
    """
    Read an st.UploadedFile safely.
    Returns (bytes, None) or (None, error_str).
    """
    try:
        data = uploaded_file.getvalue()          # works after rerun too
        if not data:
            return None, f"{uploaded_file.name}: empty file."
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_TYPES:
            return None, f"{uploaded_file.name}: unsupported type (.{ext})."
        return data, None
    except Exception as exc:
        log.error("safe_read_uploaded: %s", exc)
        return None, f"{uploaded_file.name}: read error — {exc}"


def display_image_from_bytes(img_bytes: bytes, caption: str = ""):
    """
    Show an image from raw bytes without TypeError.
    Uses PIL so the ndarray dtype is guaranteed uint8 RGB.
    """
    try:
        arr, err = bytes_to_rgb_array(img_bytes)
        if err:
            st.warning(f"Cannot display image: {err}")
            return
        st.image(arr, caption=caption, use_container_width=True)
    except Exception as exc:
        log.error("display_image_from_bytes: %s", exc)
        st.warning("Could not display image.")


# ════════════════════════════════════════════════════════
# FACE ENCODING  (mirrors modules/face_utils.py)
# ════════════════════════════════════════════════════════
# We keep ONE encoding path.
# face_recognition  → 128-D float64 vector  (preferred)
# OpenCV fallback   → 128-D float64 vector  (same shape via PCA-like resize)
# Encodings from different methods MUST NOT be mixed in the same DB.
# A "method" flag is stored alongside encodings to detect mismatches.

ENCODING_METHOD_KEY = "encoding_method"   # stored in session_state

# Sentinel error codes returned by encode_face(). The UI maps these to
# specific, non-misleading messages instead of a generic "invalid image".
ERR_NO_FACE           = "no_face"            # valid image, genuinely no face
ERR_ENGINE_UNAVAILABLE = "engine_unavailable"  # cv2 AND face_recognition missing
ERR_PROCESSING        = "processing_error"    # unexpected runtime error


def _encode_face_recognition(arr: np.ndarray):
    """face_recognition (dlib) path → 128-D vector. Assumes it's available."""
    encs = face_recognition.face_encodings(arr)
    if not encs:
        return None, ERR_NO_FACE
    return encs[0].astype(np.float64), None


def _encode_opencv(arr: np.ndarray):
    """
    OpenCV Haar-cascade fallback path. Assumes cv2 is available.
    Returns the same normalized 128x128 grayscale vector as the Flask
    OpenCV fallback. This is a lower-accuracy path than face_recognition.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    fc   = cv2.CascadeClassifier(
               cv2.data.haarcascades +
               "haarcascade_frontalface_default.xml")
    faces = fc.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, ERR_NO_FACE
    # Use largest detected face
    x, y, w, h = faces[0]
    roi  = gray[y:y+h, x:x+w]
    # Resize to 16×8 = 128 pixels (same final dim as face_recognition)
    small = cv2.resize(roi, (128, 128))
    flat  = small.flatten().astype(np.float64)
    norm  = np.linalg.norm(flat)
    return (flat / norm if norm > 0 else flat), None


def encode_face(arr: np.ndarray):
    """
    Returns (encoding: np.ndarray[128] | None, method: str, error: str | None).

    error is None on success. Otherwise it is one of:
      ERR_NO_FACE            - a valid, readable image with no detectable face
      ERR_ENGINE_UNAVAILABLE - neither cv2 nor face_recognition could be
                                imported (a dependency/deployment problem,
                                NOT a bad photo — must never be reported to
                                the user as "invalid image")
      ERR_PROCESSING         - an unexpected runtime error while encoding

    Order of preference: face_recognition (dlib, high accuracy) first,
    OpenCV Haar-cascade fallback second. A missing dependency for one
    engine never blocks the other.
    """
    if not FACE_ENGINE_AVAILABLE:
        return None, "none", ERR_ENGINE_UNAVAILABLE

    saw_no_face = False

    if FACE_RECOGNITION_AVAILABLE:
        try:
            enc, err = _encode_face_recognition(arr)
            if err is None:
                return enc, "face_recognition", None
            saw_no_face = True
        except Exception as exc:
            log.error("face_recognition encode error: %s", exc)
            # fall through and still try the OpenCV path below

    if CV2_AVAILABLE:
        try:
            enc, err = _encode_opencv(arr)
            if err is None:
                return enc, "opencv", None
            saw_no_face = True
        except Exception as exc:
            log.error("OpenCV encode error: %s", exc)

    if saw_no_face:
        return None, "none", ERR_NO_FACE
    return None, "none", ERR_PROCESSING


# Human-readable messages shown in the UI for each error code.
ERROR_MESSAGES = {
    ERR_NO_FACE: "No face detected in the image. Please use a clear, "
                 "front-facing photo with good lighting.",
    ERR_ENGINE_UNAVAILABLE: "Face detection engine unavailable on this "
                 "server (missing dependencies: cv2/face_recognition). "
                 "This is a deployment issue, not a problem with your "
                 "photo — please contact the administrator.",
    ERR_PROCESSING: "Could not process this image due to an unexpected "
                 "error. Please try a different photo.",
}


def face_error_message(err_code: str) -> str:
    return ERROR_MESSAGES.get(err_code, f"Error: {err_code}")


def compare_encodings(stored: np.ndarray, query: np.ndarray) -> float:
    """Euclidean distance between two same-shape encodings."""
    if stored.shape != query.shape:
        log.warning("Shape mismatch: %s vs %s", stored.shape, query.shape)
        return 999.0
    return float(np.linalg.norm(stored.astype(np.float64) -
                                query.astype(np.float64)))


# ════════════════════════════════════════════════════════
# RECOGNITION  (mirrors app.py recognize_from_bytes)
# ════════════════════════════════════════════════════════
def recognize_from_bytes(img_bytes: bytes):
    """
    Returns (student_dict | None, error_str | None).
    student_dict contains 'confidence' key added here.
    """
    arr, err = bytes_to_rgb_array(img_bytes)
    if err:
        return None, err

    query_enc, method, err = encode_face(arr)
    if err:
        return None, face_error_message(err)

    db = get_db()
    students = db.execute(
        "SELECT * FROM students WHERE face_encodings IS NOT NULL"
    ).fetchall()

    if not students:
        return None, "No trained students in database yet."

    best_student  = None
    best_distance = float("inf")

    for row in students:
        try:
            stored_list = pickle.loads(row["face_encodings"])
            if not isinstance(stored_list, list) or len(stored_list) == 0:
                continue
            for stored_enc in stored_list:
                stored_arr = np.array(stored_enc, dtype=np.float64)
                dist = compare_encodings(stored_arr, query_enc)
                if dist < best_distance:
                    best_distance = dist
                    best_student  = dict(row)
        except Exception as exc:
            log.warning("Skipping student id=%s: %s", row["id"], exc)
            continue

    if best_student:
        raw_confidence = (1 - best_distance) * 100
        confidence = round(raw_confidence, 1)
        if raw_confidence < LOW_CONFIDENCE_THRESHOLD:
            return None, "The Student has not been recorded in the Database"
        if raw_confidence < MEDIUM_CONFIDENCE_THRESHOLD:
            return None, "Please Make More Clear Photo"
        best_student["confidence"] = confidence
        return best_student, None

    return None, "Same image not found in Database."


def recognize_from_bytes_once(img_bytes: bytes):
    """Reuse the current session's result for an unchanged image and model."""
    db = get_db()
    model_hash = hashlib.sha256()
    for row in db.execute(
            "SELECT id, face_encodings FROM students "
            "WHERE face_encodings IS NOT NULL ORDER BY id"):
        model_hash.update(str(row["id"]).encode())
        model_hash.update(row["face_encodings"] or b"")
    image_hash = hashlib.sha256(img_bytes).hexdigest()
    cache_key = (image_hash, model_hash.hexdigest())
    cached = st.session_state.get("recognition_result_cache")
    if cached and cached[0] == cache_key:
        return cached[1], cached[2]
    result, error = recognize_from_bytes(img_bytes)
    st.session_state.recognition_result_cache = (cache_key, result, error)
    return result, error


# ════════════════════════════════════════════════════════
# TRAINING HELPERS
# ════════════════════════════════════════════════════════
def retrain_student(student_db_id: int, db) -> tuple:
    """
    Re-encode all stored images for a student.
    Returns (success: bool, trained_count: int, skipped_count: int).

    IMPORTANT: if the face-detection engine itself is unavailable
    (dependency/deployment problem), this function does NOT touch the
    student's existing face_encodings — a temporary missing dependency
    must never wipe out previously-trained data.
    """
    if not FACE_ENGINE_AVAILABLE:
        log.error("retrain_student aborted: %s", ERR_ENGINE_UNAVAILABLE)
        return False, 0, 0

    images = db.execute(
        "SELECT id, image_data FROM face_images WHERE student_id=?",
        (student_db_id,)
    ).fetchall()

    if not images:
        db.execute(
            "UPDATE students SET face_encodings=NULL WHERE id=?",
            (student_db_id,))
        db.commit()
        return False, 0, 0

    encodings = []
    skipped   = 0

    for row in images:
        try:
            img_bytes = base64.b64decode(row["image_data"])
            arr, err  = bytes_to_rgb_array(img_bytes)
            if err:
                log.warning("retrain img id=%s: %s", row["id"], err)
                skipped += 1
                continue
            enc, method, err = encode_face(arr)
            if err:
                log.warning("retrain encode img id=%s: %s", row["id"], err)
                skipped += 1
                continue
            encodings.append(enc)
        except Exception as exc:
            log.warning("retrain exception img id=%s: %s", row["id"], exc)
            skipped += 1
            continue

    if encodings:
        db.execute(
            "UPDATE students SET face_encodings=? WHERE id=?",
            (pickle.dumps(encodings), student_db_id))
        db.commit()
        return True, len(encodings), skipped

    # No valid encodings — clear BLOB
    db.execute(
        "UPDATE students SET face_encodings=NULL WHERE id=?",
        (student_db_id,))
    db.commit()
    return False, 0, skipped


def attendance_payload(db, session_id: int):
    session_row = db.execute(
        "SELECT * FROM attendance_sessions WHERE id=?", (session_id,)
    ).fetchone()
    if not session_row:
        return None
    present = db.execute(
        """SELECT a.*, s.name, s.student_id AS external_id, s.roll_number
           FROM attendance a JOIN students s ON s.id=a.student_id
           WHERE a.session_id=?""",
        (session_id,),
    ).fetchall()
    total = db.execute(
        "SELECT COUNT(*) FROM attendance_roster WHERE session_id=?",
        (session_id,),
    ).fetchone()[0]
    absent = db.execute(
        """SELECT s.id, s.name, s.student_id AS external_id, s.roll_number
           FROM attendance_roster r JOIN students s ON s.id=r.student_id
           LEFT JOIN attendance a ON a.student_id=r.student_id AND a.session_id=r.session_id
           WHERE r.session_id=? AND a.id IS NULL""",
        (session_id,),
    ).fetchall()
    present = sort_attendance_rows([dict(x) for x in present])
    absent = sort_attendance_rows([dict(x) for x in absent])
    return {
        "session": dict(session_row), "present": present,
        "absent": absent, "total": total,
        "count": len(present),
        "percentage": round(100 * len(present) / total, 1) if total else 0,
    }


def sort_attendance_rows(rows):
    """Sort attendance records by natural roll-number order."""
    def key(row):
        parts = re.split(r"(\d+)", str(row.get("roll_number") or ""))
        natural = tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                        for part in parts if part)
        return natural, str(row.get("name") or "").casefold()
    return sorted(rows, key=key)


def attendance_csv(rows, columns):
    """Build an Excel-friendly UTF-8 CSV download for an attendance list."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def recognize_attendance_image(img_bytes: bytes, session_id: int):
    """Recognize only against the selected class roster and record one check-in."""
    arr, error = bytes_to_rgb_array(img_bytes)
    if error:
        return None, error
    query, _, error = encode_face(arr)
    if error:
        return None, face_error_message(error)
    db = get_db()
    roster = db.execute(
        """SELECT s.* FROM students s JOIN attendance_roster r ON r.student_id=s.id
           WHERE r.session_id=? AND s.face_encodings IS NOT NULL""",
        (session_id,),
    ).fetchall()
    best, distance = None, float("inf")
    for row in roster:
        try:
            for known in pickle.loads(row["face_encodings"]):
                dist = compare_encodings(np.asarray(known), query)
                if dist < distance:
                    best, distance = dict(row), dist
        except Exception:
            continue
    if best is None:
        return None, "No registered student match in this class."
    confidence = (1 - distance) * 100
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return None, "The Student has not been recorded in the Database"
    if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return None, "Please Make More Clear Photo"
    db.execute(
        """INSERT OR IGNORE INTO attendance
           (student_id,session_id,session_date,check_in_timestamp,status,recognition_confidence)
           VALUES (?,?,?,?, 'present',?)""",
        (best["id"], session_id, myanmar_today(), myanmar_now(), confidence),
    )
    db.commit()
    best["confidence"] = round(confidence, 1)
    return best, None


# ════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════
for key, val in {
    "admin_logged_in": False,
    "page":            "home",
    "admin_return_page": None,
    "edit_sid":        None,
    "train_sid":       None,
    "train_done":      False,
    "attendance_session_id": None,
    "attendance_capture_id": None,
    "attendance_notice": None,
    "delete_student_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


def go(page: str, **kwargs):
    st.session_state.page = page
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()


# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
# Main navigation routes protected pages through the existing login form.
def navigate_to(destination: str, requires_admin: bool = False):
    if requires_admin and not st.session_state.admin_logged_in:
        st.session_state.admin_return_page = destination
        go("login")
    go(destination)


page = st.session_state.page


# ════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════
if page == "home":
    st.markdown("""
    <div class='hero'>
      <div style='font-size:3rem'>🎓</div>
      <h1>PTU Student Face Recognition System</h1>
      <p>Pyay Technological University — Smart Attendance &amp; Student Management</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='ptu-card ptu-card-blue'>
          <h4>🤖 Face Detection</h4>
          <p style='color:#555;font-size:.9rem'>
          AI-powered recognition using OpenCV and face_recognition library.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='ptu-card ptu-card-green'>
          <h4>💾 Student Database</h4>
          <p style='color:#555;font-size:.9rem'>
          All records stored securely in local SQLite database.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='ptu-card ptu-card-warn'>
          <h4>🧠 AI Training</h4>
          <p style='color:#555;font-size:.9rem'>
          Train face models with minimum 2 images per student.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Search Student by ID",
                     use_container_width=True, type="primary"):
            go("search")
    with col2:
        if st.button("📷 Try Face Detection", use_container_width=True):
            go("detect")

    st.markdown("### Find a student")
    with st.form("home_student_search_form"):
        home_student_id = st.text_input("Student ID", placeholder="e.g. PTU-2024-001")
        home_search = st.form_submit_button("Search student", type="primary", use_container_width=True)
    if home_search:
        lookup_id = home_student_id.strip()
        if not lookup_id:
            st.error("Enter a Student ID to search.")
        else:
            student = get_db().execute(
                "SELECT name,student_id,major,semester,roll_number FROM students WHERE student_id=?",
                (lookup_id,),
            ).fetchone()
            if student:
                st.success(f"Student found: {student['name']}")
                info_cols = st.columns(2)
                info_cols[0].write(f"**Student ID:** {student['student_id']}\n\n**Roll number:** {student['roll_number']}")
                info_cols[1].write(f"**Major:** {student['major']}\n\n**Semester:** {student['semester']}")
            else:
                st.error(f"No student found with ID: {lookup_id}")


# ════════════════════════════════════════════════════════
# PAGE: SEARCH
# ════════════════════════════════════════════════════════
elif page == "search":
    st.markdown("## 🔍 Search Student by ID")
    st.caption("Enter a Student ID to retrieve the complete student profile")

    with st.form("search_form"):
        sid_input = st.text_input(
            "Student ID", placeholder="e.g. PTU-2024-001",
            label_visibility="collapsed")
        submitted = st.form_submit_button(
            "🔍 Search", type="primary", use_container_width=True)

    if submitted:
        sid_input = sid_input.strip()
        if not sid_input:
            st.error("⚠️ Please enter a Student ID.")
        else:
            try:
                db = get_db()
                row = db.execute(
                    "SELECT * FROM students WHERE student_id=?",
                    (sid_input,)).fetchone()
                if row:
                    s = dict(row)
                    st.success("✅ Student found!")
                    st.markdown(f"""
                    <div class='result-card'>
                      <h3>👤 {s['name']}</h3>
                      <span class='badge-success'>{s['major']}</span>
                      <div class='info-row'>
                        <div class='info-box'>
                          <small>🪪 Student ID</small>
                          <b style='color:#0d6efd'>{s['student_id']}</b>
                        </div>
                        <div class='info-box'>
                          <small>🔢 Roll Number</small><b>{s['roll_number']}</b>
                        </div>
                        <div class='info-box'>
                          <small>🏛️ Major</small><b>{s['major']}</b>
                        </div>
                        <div class='info-box'>
                          <small>📚 Semester</small><b>{s['semester']}</b>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ No student found with ID: **{sid_input}**")
            except Exception as exc:
                log.error("search: %s", exc)
                st.error(f"Database error: {exc}")


# ════════════════════════════════════════════════════════
# PAGE: FACE DETECT
# ════════════════════════════════════════════════════════
elif page == "detect":
    st.markdown("## 📷 Face Detection & Recognition")
    st.caption("Upload a photo or use camera to identify a student")

    if not FACE_ENGINE_AVAILABLE:
        st.error(
            "🚫 **Face detection engine unavailable on this server.** "
            "Neither `cv2` (OpenCV) nor `face_recognition` could be "
            "imported. This is a deployment/dependency problem — "
            "please contact the administrator.")
        with st.expander("🔧 Technical details"):
            st.json(face_engine_status())
    elif not FACE_RECOGNITION_AVAILABLE:
        st.warning(
            "⚠️ Running in **OpenCV-only fallback mode** — recognition "
            "accuracy is reduced because `face_recognition`/`dlib` is "
            "not available.")

    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

    # ── Upload tab ───────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload face image",
            type=ALLOWED_TYPES,
            label_visibility="collapsed",
            key="detect_upload"
        )
        if uploaded is not None:
            img_bytes, err = safe_read_uploaded(uploaded)
            if err:
                st.error(f"❌ {err}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    display_image_from_bytes(img_bytes, "Uploaded Image")
                with col2:
                    with st.spinner("🔍 Detecting and recognizing face…"):
                        result, rec_err = recognize_from_bytes_once(img_bytes)
                    if result:
                        st.success(
                            f"✅ Student Detected!  "
                            f"Confidence: **{result['confidence']}%**")
                        st.markdown(f"""
                        <div class='result-card'>
                          <h3>Detected Student:</h3>
                          <div class='info-row'>
                            <div class='info-box'>
                              <small>Student Name</small>
                              <b>{result['name']}</b></div>
                            <div class='info-box'>
                              <small>Student ID</small>
                              <b style='color:#0d6efd'>
                              {result['student_id']}</b></div>
                            <div class='info-box'>
                              <small>Major</small>
                              <b>{result['major']}</b></div>
                            <div class='info-box'>
                              <small>Semester</small>
                              <b>{result['semester']}</b></div>
                            <div class='info-box'>
                              <small>Roll Number</small>
                              <b>{result['roll_number']}</b></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(result["confidence"] / 100)
                    else:
                        st.error(f"❌ {rec_err}")

    # ── Camera tab ───────────────────────────────────────
    with tab_camera:
        st.info("Allow camera access when prompted. On a phone, this opens the phone camera in your browser; the photo is uploaded to this Streamlit server for recognition. If camera access is unavailable, use Upload Image and choose Take Photo from your phone's image picker.")
        camera_img = st.camera_input(
            "Take a photo", key="detect_camera")

        if camera_img is not None:
            try:
                img_bytes = camera_img.getvalue()
                if not img_bytes:
                    st.error("❌ Camera returned an empty image. Please try again.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        display_image_from_bytes(img_bytes, "Captured Image")
                    with col2:
                        with st.spinner("🔍 Detecting and recognizing face…"):
                            result, rec_err = recognize_from_bytes_once(img_bytes)
                        if result:
                            st.success(
                                f"✅ Student Detected!  "
                                f"Confidence: **{result['confidence']}%**")
                            st.markdown(f"""
                            <div class='result-card'>
                              <h3>Detected Student:</h3>
                              <div class='info-row'>
                                <div class='info-box'>
                                  <small>Student Name</small>
                                  <b>{result['name']}</b></div>
                                <div class='info-box'>
                                  <small>Student ID</small>
                                  <b style='color:#0d6efd'>
                                  {result['student_id']}</b></div>
                                <div class='info-box'>
                                  <small>Major</small>
                                  <b>{result['major']}</b></div>
                                <div class='info-box'>
                                  <small>Semester</small>
                                  <b>{result['semester']}</b></div>
                                <div class='info-box'>
                                  <small>Roll Number</small>
                                  <b>{result['roll_number']}</b></div>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(result["confidence"] / 100)
                        else:
                            st.error(f"❌ {rec_err}")
            except Exception as exc:
                log.error("camera detect: %s", exc)
                st.error(f"Camera processing error: {exc}")


# ════════════════════════════════════════════════════════
# PAGE: ADMIN LOGIN
# ════════════════════════════════════════════════════════
elif page == "login":
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div class='ptu-card' style='text-align:center;padding:2rem'>
          <div style='font-size:2.5rem'>🔐</div>
          <h3 style='color:#0f3460'>Administrator Login</h3>
          <p style='color:#666;font-size:.9rem'>PTU Face Recognition System</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username  = st.text_input("Username", placeholder=ADMIN_USERNAME)
            password  = st.text_input("Password", type="password",
                                    placeholder="••••••••")
            submitted = st.form_submit_button(
                "🔐 Login", type="primary", use_container_width=True)

        if submitted:
            if (username == ADMIN_USERNAME and
                    password == ADMIN_PASSWORD):
                st.session_state.admin_logged_in = True
                destination = st.session_state.pop("admin_return_page", None)
                if destination not in {"dashboard", "students", "training", "add", "attendance"}:
                    destination = "dashboard"
                go(destination)
            else:
                st.error("❌ Invalid credentials. Please try again.")


# ════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════
elif page == "dashboard":
    if not st.session_state.admin_logged_in:
        go("login")

    st.markdown("## 📊 Admin Dashboard")
    if st.button("Log out", key="dashboard_logout"):
        st.session_state.admin_logged_in = False
        go("home")
    db      = get_db()
    total   = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    trained = db.execute(
        "SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL"
    ).fetchone()[0]
    pending = total - trained

    today = myanmar_today()
    active_session = db.execute(
        "SELECT id,session_name FROM attendance_sessions WHERE status='active' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    today_session = db.execute(
        "SELECT id,session_name,status FROM attendance_sessions WHERE session_date=? ORDER BY id DESC LIMIT 1",
        (today,),
    ).fetchone()
    if today_session:
        present_today = db.execute(
            "SELECT COUNT(*) FROM attendance WHERE session_id=?", (today_session["id"],)
        ).fetchone()[0]
        roster_today = db.execute(
            "SELECT COUNT(*) FROM attendance_roster WHERE session_id=?", (today_session["id"],)
        ).fetchone()[0]
        absent_today = max(roster_today - present_today, 0)
        attendance_pct = round(100 * present_today / roster_today, 1) if roster_today else 0
        session_label = f"{today_session['session_name']} · {today_session['status'].title()}"
    else:
        present_today = absent_today = attendance_pct = 0
        session_label = "No session today"

    st.markdown("### Today's attendance")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Registered", total)
    d2.metric("Present", present_today)
    d3.metric("Absent", absent_today)
    d4.metric("Attendance", f"{attendance_pct}%")
    d5.metric("Session status", "Active" if active_session else ("Ended" if today_session else "Not started"))
    st.caption(f"{session_label} · Myanmar Standard Time")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='stat-card' style='background:linear-gradient("
            f"135deg,#667eea,#764ba2)'>"
            f"<div class='stat-num'>{total}</div>"
            f"<div class='stat-lbl'>👥 Total Students</div></div>",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='stat-card' style='background:linear-gradient("
            f"135deg,#11998e,#38ef7d)'>"
            f"<div class='stat-num'>{trained}</div>"
            f"<div class='stat-lbl'>✅ Trained</div></div>",
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"<div class='stat-card' style='background:linear-gradient("
            f"135deg,#f093fb,#f5576c)'>"
            f"<div class='stat-num'>{pending}</div>"
            f"<div class='stat-lbl'>⏳ Pending</div></div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Quick Actions")
    qa1, qa2, qa3, qa4, qa5 = st.columns(5)
    with qa1:
        if st.button("➕ Add Student",
                     use_container_width=True, type="primary"):
            go("add")
    with qa2:
        if st.button("👥 View All", use_container_width=True):
            go("students")
    with qa3:
        if st.button("🔄 Retrain All", use_container_width=True):
            if not FACE_ENGINE_AVAILABLE:
                st.error(face_error_message(ERR_ENGINE_UNAVAILABLE))
            else:
                with st.spinner("Retraining all students…"):
                    rows  = db.execute("SELECT id FROM students").fetchall()
                    count = 0
                    for r in rows:
                        ok, _, _ = retrain_student(r["id"], db)
                        if ok:
                            count += 1
                st.success(f"✅ Retrained {count} / {len(rows)} students.")
    with qa4:
        if st.button("📷 Test Detect", use_container_width=True):
            go("detect")
    with qa5:
        if st.button("📋 Attendance", use_container_width=True):
            go("attendance")

    st.markdown("#### 🕐 Recently Added Students")
    recent = db.execute(
        "SELECT * FROM students ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    if recent:
        for row in recent:
            s     = dict(row)
            badge = ("✅ Trained" if s["face_encodings"]
                     else "⏳ Pending")
            c1, c2, c3, c4, c5 = st.columns([3,2,3,2,2])
            c1.write(f"**{s['name']}**")
            c2.write(f"`{s['student_id']}`")
            c3.write(s["major"])
            c4.write(s["semester"])
            c5.write(badge)
    else:
        st.info("No students added yet.")

    with st.expander("🔧 System diagnostics (face detection engine)"):
        status = face_engine_status()
        d1, d2 = st.columns(2)
        with d1:
            st.write("**OpenCV (cv2)**")
            if status["cv2_available"]:
                st.success(f"✅ Available — v{status['cv2_version']}")
            else:
                st.error(f"❌ Unavailable — {status['cv2_error']}")
        with d2:
            st.write("**face_recognition (dlib)**")
            if status["face_recognition_available"]:
                st.success("✅ Available")
            else:
                st.error(f"❌ Unavailable — {status['face_recognition_error']}")
        if not status["any_engine_available"]:
            st.error(
                "No face-detection engine is available. Check "
                "`requirements.txt` is named exactly that (Streamlit "
                "Cloud only auto-installs from `requirements.txt`), and "
                "that `packages.txt` includes the system build "
                "dependencies for dlib/OpenCV.")


# ════════════════════════════════════════════════════════
# PAGE: STUDENTS LIST
# ════════════════════════════════════════════════════════
elif page == "students":
    if not st.session_state.admin_logged_in:
        go("login")

    st.markdown("## 👥 All Students")
    db       = get_db()
    students = db.execute(
        "SELECT * FROM students ORDER BY created_at DESC"
    ).fetchall()
    st.caption(f"{len(students)} total records")

    search = st.text_input(
        "Search", placeholder="🔍 Search by name, ID, or major…",
        label_visibility="collapsed")

    for row in students:
        s = dict(row)
        if search and search.lower() not in (
                s["name"] + s["student_id"] + s["major"]).lower():
            continue

        icon = "✅" if s["face_encodings"] else "⏳"
        with st.expander(f"{icon}  {s['name']}  —  `{s['student_id']}`"):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Major**<br>{s['major']}",
                        unsafe_allow_html=True)
            c2.markdown(f"**Semester**<br>{s['semester']}",
                        unsafe_allow_html=True)
            c3.markdown(f"**Roll No.**<br>{s['roll_number']}",
                        unsafe_allow_html=True)
            img_n = db.execute(
                "SELECT COUNT(*) FROM face_images WHERE student_id=?",
                (s["id"],)).fetchone()[0]
            c4.markdown(f"**Images**<br>{img_n} uploaded",
                        unsafe_allow_html=True)

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🧠 Train",
                             key=f"tr_{s['id']}",
                             use_container_width=True):
                    go("train", train_sid=s["id"], train_done=False)
            with b2:
                if st.button("✏️ Edit",
                             key=f"ed_{s['id']}",
                             use_container_width=True):
                    go("edit", edit_sid=s["id"])
            with b3:
                if st.button("🗑️ Delete",
                             key=f"dl_{s['id']}",
                             use_container_width=True):
                    st.session_state.delete_student_id = s["id"]
                    st.rerun()
            if st.session_state.delete_student_id == s["id"]:
                st.warning(f"Delete {s['name']} and its stored face images? This cannot be undone.")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    confirm_delete = st.button("Confirm delete", key=f"confirm_dl_{s['id']}", type="primary", use_container_width=True)
                with cancel_col:
                    cancel_delete = st.button("Cancel", key=f"cancel_dl_{s['id']}", use_container_width=True)
                if cancel_delete:
                    st.session_state.delete_student_id = None
                    st.rerun()
                if confirm_delete:
                    try:
                        db.execute(
                            "DELETE FROM face_images WHERE student_id=?",
                            (s["id"],))
                        db.execute(
                            "DELETE FROM students WHERE id=?",
                            (s["id"],))
                        db.commit()
                        st.session_state.delete_student_id = None
                        st.success(f"Deleted {s['name']}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete error: {exc}")


# ════════════════════════════════════════════════════════
# PAGE: ADD STUDENT
# ════════════════════════════════════════════════════════
elif page == "add":
    if not st.session_state.admin_logged_in:
        go("login")

    st.markdown("## ➕ Add New Student")

    with st.form("add_form", clear_on_submit=True):
        name    = st.text_input("Student Name *")
        c1, c2  = st.columns(2)
        sid_in  = c1.text_input("Student ID *", placeholder="PTU-2024-001")
        roll    = c2.text_input("Roll Number *")
        major   = st.selectbox("Major *", MAJORS)
        sem     = st.selectbox("Semester *", SEMESTERS)
        ok_btn  = st.form_submit_button(
            "✅ Add Student", type="primary", use_container_width=True)

    if ok_btn:
        if not all([name.strip(), sid_in.strip(), roll.strip()]):
            st.error("❌ All fields are required.")
        else:
            try:
                db = get_db()
                db.execute(
                    "INSERT INTO students "
                    "(name,student_id,major,semester,roll_number) "
                    "VALUES (?,?,?,?,?)",
                    (name.strip(), sid_in.strip(), major, sem, roll.strip()))
                db.commit()
                st.success(f"✅ **{name.strip()}** added successfully!")
                st.info("👉 Go to Students → 🧠 Train to upload face images.")
            except sqlite3.IntegrityError:
                st.error(f"❌ Student ID **{sid_in.strip()}** already exists.")
            except Exception as exc:
                log.error("add student: %s", exc)
                st.error(f"Database error: {exc}")


# ════════════════════════════════════════════════════════
# PAGE: EDIT STUDENT
# ════════════════════════════════════════════════════════
elif page == "edit":
    if not st.session_state.admin_logged_in:
        go("login")

    sid = st.session_state.get("edit_sid")
    if not sid:
        go("students")

    db  = get_db()
    row = db.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()
    if not row:
        st.error("Student not found.")
        if st.button("← Back"):
            go("students")
        st.stop()

    s = dict(row)
    st.markdown(f"## ✏️ Edit Student — {s['name']}")

    with st.form("edit_form"):
        name  = st.text_input("Student Name", value=s["name"])
        c1,c2 = st.columns(2)
        sid_v = c1.text_input("Student ID",   value=s["student_id"])
        roll  = c2.text_input("Roll Number",  value=s["roll_number"])
        edit_majors = MAJORS if s["major"] in MAJORS else [s["major"], *MAJORS]
        edit_semesters = SEMESTERS if s["semester"] in SEMESTERS else [s["semester"], *SEMESTERS]
        maj_i = edit_majors.index(s["major"])
        sem_i = edit_semesters.index(s["semester"])
        major = st.selectbox("Major",    edit_majors,    index=maj_i)
        sem   = st.selectbox("Semester", edit_semesters, index=sem_i)
        b1,b2 = st.columns(2)
        save  = b1.form_submit_button(
            "💾 Save", type="primary", use_container_width=True)
        back  = b2.form_submit_button(
            "Cancel", use_container_width=True)

    if save:
        if not all((name.strip(), sid_v.strip(), roll.strip())):
            st.error("All fields are required.")
        else:
            try:
                db.execute(
                    "UPDATE students SET name=?,student_id=?,major=?,"
                    "semester=?,roll_number=? WHERE id=?",
                    (name.strip(), sid_v.strip(), major, sem,
                     roll.strip(), sid))
                db.commit()
                st.success("✅ Updated successfully!")
                go("students")
            except sqlite3.IntegrityError:
                st.error(f"❌ Student ID **{sid_v.strip()}** already exists.")
            except Exception as exc:
                log.exception("edit student failed")
                st.error(f"Update error: {exc}")
    if back:
        go("students")


# ════════════════════════════════════════════════════════
# PAGE: TRAIN FACES
# ════════════════════════════════════════════════════════
elif page == "training":
    if not st.session_state.admin_logged_in:
        st.session_state.admin_return_page = "training"
        go("login")

    st.markdown("## 🧠 AI Training")
    st.caption("Choose a registered student to add training photos or rebuild their face encodings.")
    db = get_db()
    training_students = db.execute(
        "SELECT id,name,student_id FROM students ORDER BY name COLLATE NOCASE"
    ).fetchall()
    if not training_students:
        st.info("There are no registered students yet. Add a student before training a face model.")
        if st.button("Add a student", type="primary", use_container_width=True):
            go("add")
    else:
        training_options = {
            f"{student['name']} ({student['student_id']})": student["id"]
            for student in training_students
        }
        with st.form("training_student_picker"):
            selected_student = st.selectbox("Student", list(training_options))
            open_training = st.form_submit_button(
                "Open training workspace", type="primary", use_container_width=True
            )
        if open_training:
            go("train", train_sid=training_options[selected_student], train_done=False)

elif page == "train":
    if not st.session_state.admin_logged_in:
        go("login")

    sid = st.session_state.get("train_sid")
    if not sid:
        go("students")

    db  = get_db()
    row = db.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()
    if not row:
        st.error("Student not found.")
        if st.button("← Back"):
            go("students")
        st.stop()

    s = dict(row)
    st.markdown(f"## 🧠 Train Faces — {s['name']}")

    col_info, col_upload = st.columns([1, 2])

    # ── Student info card ────────────────────────────────
    with col_info:
        img_n = db.execute(
            "SELECT COUNT(*) FROM face_images WHERE student_id=?",
            (sid,)).fetchone()[0]

        st.markdown(f"""
        <div class='ptu-card ptu-card-blue' style='text-align:center'>
          <div style='width:68px;height:68px;border-radius:50%;
               background:linear-gradient(135deg,#667eea,#764ba2);
               display:flex;align-items:center;justify-content:center;
               color:#fff;font-size:1.8rem;margin:0 auto .7rem'>
            {s['name'][0].upper()}
          </div>
          <h4 style='margin:.2rem 0'>{s['name']}</h4>
          <code>{s['student_id']}</code><br><br>
          <span class='badge-success'>{s['major']}</span><br><br>
          <small>Semester: {s['semester']}</small><br>
          <small>Roll No.: {s['roll_number']}</small>
        </div>
        """, unsafe_allow_html=True)

        if s["face_encodings"]:
            st.success("✅ Face Trained")
        else:
            st.warning("⏳ Not Trained Yet")

        st.metric("Training Images", img_n,
                  delta="need ≥ 2" if img_n < 2 else "ready")

        if st.button("← Back to Students",
                     use_container_width=True):
            go("students")

    # ── Upload section ───────────────────────────────────
    with col_upload:
        st.markdown("#### 📤 Upload Face Images")

        if not FACE_ENGINE_AVAILABLE:
            st.error(
                "🚫 **Face detection engine unavailable on this server.** "
                "Neither `cv2` (OpenCV) nor `face_recognition` could be "
                "imported. This is a deployment/dependency problem, not "
                "an issue with your photos — uploads are disabled until "
                "it's fixed. Contact the administrator.")
            with st.expander("🔧 Technical details"):
                st.json(face_engine_status())
        else:
            st.info(
                "Upload **at least 2** clear, front-facing photos. "
                "Multiple images improve recognition accuracy.")
            if not FACE_RECOGNITION_AVAILABLE:
                st.warning(
                    "⚠️ Running in **OpenCV-only fallback mode** — the "
                    "high-accuracy `face_recognition`/`dlib` engine is "
                    "not available, so matches will be less reliable. "
                    "See the technical note in the sidebar/deployment "
                    "docs to enable it.")

        files = st.file_uploader(
            "Choose images",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"train_upload_{sid}"      # unique key per student
        )

        # Preview thumbnails
        if files:
            thumb_cols = st.columns(min(len(files), 5))
            for i, f in enumerate(files):
                with thumb_cols[i % 5]:
                    try:
                        thumb_bytes = f.getvalue()
                        arr, err = bytes_to_rgb_array(thumb_bytes)
                        if arr is not None:
                            st.image(arr,
                                     caption=f.name[:12],
                                     use_container_width=True)
                        else:
                            st.caption(f"❌ {f.name[:12]}")
                    except Exception:
                        st.caption(f"❌ {f.name[:12]}")

        train_btn = st.button(
            "🧠 Upload & Train",
            type="primary",
            use_container_width=True,
            disabled=(not files or not FACE_ENGINE_AVAILABLE)
        )

        if train_btn and files and not st.session_state.train_done:
            if not FACE_ENGINE_AVAILABLE:
                # Defensive guard — the button should already be disabled,
                # but never let a dependency outage masquerade as "bad photos".
                st.error(face_error_message(ERR_ENGINE_UNAVAILABLE))
                st.stop()

            progress   = st.progress(0, text="Processing images…")
            total_f    = len(files)
            saved      = 0
            skipped    = 0
            skip_msgs  = []

            for i, f in enumerate(files):
                progress.progress(
                    (i + 1) / total_f,
                    text=f"Processing {f.name} ({i+1}/{total_f})…")

                img_bytes, err = safe_read_uploaded(f)
                if err:
                    skip_msgs.append(f"⚠️ {err}")
                    skipped += 1
                    continue

                arr, err = bytes_to_rgb_array(img_bytes)
                if err:
                    skip_msgs.append(f"⚠️ {f.name}: {err}")
                    skipped += 1
                    continue

                enc, method, err = encode_face(arr)
                if err:
                    skip_msgs.append(f"⚠️ {f.name}: {face_error_message(err)}")
                    skipped += 1
                    continue

                # All good — save to DB
                try:
                    b64 = base64.b64encode(img_bytes).decode()
                    db.execute(
                        "INSERT INTO face_images "
                        "(student_id, image_data) VALUES (?,?)",
                        (sid, b64))
                    saved += 1
                except Exception as exc:
                    skip_msgs.append(f"⚠️ {f.name}: DB error — {exc}")
                    skipped += 1

            db.commit()
            progress.empty()

            # Summary
            st.info(
                f"📊 **Summary:** {total_f} uploaded → "
                f"{saved} valid → {skipped} skipped")
            if skip_msgs:
                with st.expander("⚠️ Skipped files detail"):
                    for m in skip_msgs:
                        st.write(m)

            if saved > 0:
                with st.spinner("Building face encodings…"):
                    ok, n_enc, n_skip = retrain_student(sid, db)
                if ok:
                    st.success(
                        f"✅ Training complete! "
                        f"{n_enc} encoding(s) built, {n_skip} image(s) skipped.")
                    st.session_state.train_done = True
                    st.rerun()
                else:
                    st.error(
                        "❌ Training failed — no valid face encodings "
                        "could be generated. Please use clearer, "
                        "front-facing photos.")
            else:
                st.error(
                    "❌ No valid images were saved. "
                    "Please upload clear front-facing face photos.")

        # ── Existing images gallery ──────────────────────
        st.markdown("#### 🖼️ Existing Training Images")
        existing = db.execute(
            "SELECT id, image_data FROM face_images "
            "WHERE student_id=? ORDER BY id",
            (sid,)).fetchall()

        if existing:
            g_cols = st.columns(min(len(existing), 4))
            for i, img_row in enumerate(existing):
                with g_cols[i % 4]:
                    try:
                        ib  = base64.b64decode(img_row["image_data"])
                        arr, err = bytes_to_rgb_array(ib)
                        if arr is not None:
                            st.image(arr,
                                    caption=f"Image {i+1}", 
                                        use_container_width=True)
                        else:
                            st.caption(f"Image {i+1}: corrupted")
                    except Exception as exc:
                        st.caption(f"Image {i+1}: error")
                        log.warning("gallery img %s: %s",
                                    img_row["id"], exc)

                    if st.button("🗑️ Remove",
                                 key=f"rm_{img_row['id']}",
                                 use_container_width=True):
                        db.execute(
                            "DELETE FROM face_images WHERE id=?",
                            (img_row["id"],))
                        db.commit()
                        retrain_student(sid, db)
                        st.session_state.train_done = False
                        st.rerun()
        else:
            st.info("No training images yet. Upload images above.")

        # Reset train_done so next upload session works
        if st.session_state.train_done:
            st.session_state.train_done = False


# ════════════════════════════════════════════════════════
# FALLBACK
# ════════════════════════════════════════════════════════
elif page == "attendance":
    if not st.session_state.admin_logged_in:
        go("login")

    st.markdown("## 📋 Classroom Attendance")
    st.info("Camera images are processed by this Streamlit server. Attendance dates and times are recorded in Myanmar Standard Time (MMT, UTC+06:30). Review recognition results before relying on attendance records.")
    db = get_db()
    active = db.execute(
        "SELECT id FROM attendance_sessions WHERE status='active' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if active:
        st.session_state.attendance_session_id = active["id"]

    with st.expander("Start an attendance session", expanded=not bool(active)):
        with st.form("attendance_start_form"):
            session_name = st.text_input("Class / session name")
            major = st.selectbox("Major", MAJORS, key="attendance_major")
            semester = st.selectbox("Semester", SEMESTERS[1:], key="attendance_semester")
            start = st.form_submit_button("Start Attendance", type="primary")
        if start:
            roster = db.execute(
                "SELECT id FROM students WHERE major=? AND semester=?", (major, semester)
            ).fetchall()
            if not roster:
                st.error("No students are enrolled in that major and semester.")
            elif active:
                st.error("An attendance session is already active. End it before starting another.")
            else:
                cur = db.execute(
                    """INSERT INTO attendance_sessions
                       (session_name,major,semester,session_date,started_at,status)
                       VALUES (?,?,?,?,?,'active')""",
                    (session_name.strip()[:100] or f"{major} {semester}", major, semester,
                     myanmar_today(), myanmar_now()),
                )
                session_id = cur.lastrowid
                db.executemany(
                    "INSERT INTO attendance_roster(session_id,student_id) VALUES (?,?)",
                    [(session_id, student["id"]) for student in roster],
                )
                db.commit()
                st.session_state.attendance_session_id = session_id
                st.session_state.attendance_capture_id = None
                st.rerun()

    sessions = db.execute(
        "SELECT id,session_name,session_date,status FROM attendance_sessions ORDER BY id DESC"
    ).fetchall()
    if not sessions:
        st.info("No attendance sessions yet.")
    else:
        options = {int(row["id"]): f"{row['session_name']} · {row['session_date']} · {row['status']}" for row in sessions}
        selected = st.selectbox(
            "Attendance session", list(options),
            index=(list(options).index(st.session_state.attendance_session_id)
                   if st.session_state.attendance_session_id in options else 0),
            format_func=lambda key: options[key],
        )
        st.session_state.attendance_session_id = selected
        state = attendance_payload(db, selected)
        if state:
            current = state["session"]
            st.caption(f"{current['major']} · Semester {current['semester']} · {current['session_date']} · {current['status']}")
            if current["status"] == "active":
                if st.button("Stop Attendance", type="secondary"):
                    db.execute("UPDATE attendance_sessions SET status='ended',ended_at=? WHERE id=?", (myanmar_now(), selected))
                    db.commit()
                    st.rerun()
                if not FACE_ENGINE_AVAILABLE:
                    st.error(face_error_message(ERR_ENGINE_UNAVAILABLE))
                else:
                    st.caption("This opens the camera on your phone or browser. The captured photo is sent to this Streamlit server for recognition; OpenCV cannot access your phone camera directly from the Cloud server.")
                    capture = st.camera_input("Capture a student for attendance", key=f"attendance_camera_{selected}")
                    if capture is not None:
                        raw = capture.getvalue()
                        capture_id = hashlib.sha256(raw).hexdigest()
                        if capture_id != st.session_state.attendance_capture_id:
                            st.session_state.attendance_capture_id = capture_id
                            with st.spinner("Recognizing student..."):
                                recognized, recognition_error = recognize_attendance_image(raw, selected)
                            if recognized:
                                st.session_state.attendance_notice = (
                                    "success", f"Checked in {recognized['name']} ({recognized['student_id']}) · {recognized['confidence']}% confidence"
                                )
                            else:
                                st.session_state.attendance_notice = ("error", recognition_error)
                            st.rerun()
                    notice = st.session_state.attendance_notice
                    if notice:
                        getattr(st, notice[0])(notice[1])
                        st.session_state.attendance_notice = None
            c1, c2, c3 = st.columns(3)
            c1.metric("Present", state["count"])
            c2.metric("Roster", state["total"])
            c3.metric("Attendance", f"{state['percentage']}%")

            st.markdown("### Present students · sorted by roll number")
            if state["present"]:
                for r in state["present"]:
                    with st.expander(f"Roll {r['roll_number']} · {r['name']} · {r['external_id']}"):
                        st.write(f"**Check-in:** {r['check_in_timestamp']}")
                        st.write(f"**Recognition confidence:** {r['recognition_confidence']:.1f}%")
                undo_student = st.selectbox(
                    "Undo a check-in", [r["student_id"] for r in state["present"]],
                    format_func=lambda sid: next(r["name"] + " (" + r["external_id"] + ")" for r in state["present"] if r["student_id"] == sid),
                    key=f"attendance_undo_{selected}",
                )
                if st.button("Mark selected student absent", key=f"undo_btn_{selected}"):
                    db.execute("DELETE FROM attendance WHERE session_id=? AND student_id=?", (selected, undo_student))
                    db.commit()
                    st.rerun()
            else:
                st.write("No students checked in yet.")
            present_sheet = [
                {"Roll Number": r["roll_number"], "Student Name": r["name"],
                 "Student ID": r["external_id"],
                 "Check-in Time": r["check_in_timestamp"],
                 "Confidence (%)": f"{r['recognition_confidence']:.1f}"}
                for r in state["present"]
            ]
            st.download_button(
                "Download Present Students Sheet (CSV)",
                data=attendance_csv(present_sheet, ["Roll Number", "Student Name", "Student ID", "Check-in Time", "Confidence (%)"]),
                file_name=f"attendance_{selected}_present.csv",
                mime="text/csv",
                key=f"download_present_{selected}",
                use_container_width=True,
            )

            absent_heading = "Final absent students" if current["status"] == "ended" else "Not yet checked in"
            st.markdown(f"### {absent_heading} · sorted by roll number")
            if state["absent"]:
                for r in state["absent"]:
                    with st.expander(f"Roll {r['roll_number']} · {r['name']} · {r['external_id']}"):
                        st.write("**Status:** Absent" if current["status"] == "ended" else "**Status:** Not yet checked in")
            else:
                st.write("No students are absent / waiting to check in.")
            absent_sheet = [
                {"Roll Number": r["roll_number"], "Student Name": r["name"],
                 "Student ID": r["external_id"],
                 "Status": "Absent" if current["status"] == "ended" else "Not yet checked in"}
                for r in state["absent"]
            ]
            st.download_button(
                "Download Absent Students Sheet (CSV)",
                data=attendance_csv(absent_sheet, ["Roll Number", "Student Name", "Student ID", "Status"]),
                file_name=f"attendance_{selected}_absent.csv",
                mime="text/csv",
                key=f"download_absent_{selected}",
                use_container_width=True,
            )

            if current["status"] == "ended":
                absent_rows = db.execute(
                    """SELECT s.id,s.name,s.student_id,s.roll_number FROM attendance_roster r
                       JOIN students s ON s.id=r.student_id LEFT JOIN attendance a
                       ON a.student_id=r.student_id AND a.session_id=r.session_id
                       WHERE r.session_id=? AND a.id IS NULL""", (selected,)
                ).fetchall()
                absent_rows = sort_attendance_rows([dict(r) for r in absent_rows])
                if absent_rows:
                    mark_id = st.selectbox(
                        "Correct an absent student to present", [r["id"] for r in absent_rows],
                        format_func=lambda sid: next(r["name"] + " (" + r["student_id"] + ")" for r in absent_rows if r["id"] == sid),
                        key=f"attendance_present_{selected}",
                    )
                    if st.button("Mark selected student present", key=f"present_btn_{selected}"):
                        db.execute(
                            """INSERT OR IGNORE INTO attendance
                               (student_id,session_id,session_date,check_in_timestamp,status,recognition_confidence)
                               VALUES (?,?,?,?, 'present',100)""",
                            (mark_id, selected, current["session_date"], myanmar_now()),
                        )
                        db.commit()
                        st.rerun()
                if st.button("Export to Google Sheets", key=f"export_{selected}"):
                    try:
                        import app as flask_app
                        from sheets_service import export_attendance_session
                        with flask_app.app.app_context():
                            exported = export_attendance_session(selected)
                        st.success(f"Exported {exported} new attendance row(s).")
                    except RuntimeError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        log.exception("Google Sheets attendance export failed")
                        st.error(f"Google Sheets export failed: {exc}")
else:
    go("home")


# Persistent navigation section on every rendered page. Authentication is
# checked by navigate_to before any administrator-only page is entered.
st.markdown("<div class='bottom-nav-heading'>Navigate</div>", unsafe_allow_html=True)
pending_destination = st.session_state.get("admin_return_page")
navigation = [
    ("🏠 Home", "home", False, page == "home"),
    ("👤 Admin", "dashboard", True,
     page in {"login", "dashboard", "students", "add", "edit", "training", "train", "attendance"}
     and pending_destination not in {"students", "training", "attendance"}),
    ("📷 Face Detection", "detect", False, page == "detect"),
    ("🗄️ Student Database", "students", True,
     page in {"students", "add", "edit"} or (page == "login" and pending_destination == "students")),
    ("🧠 AI Training", "training", True,
     page in {"training", "train"} or (page == "login" and pending_destination == "training")),
    ("📋 Attendance", "attendance", True,
     page == "attendance" or (page == "login" and pending_destination == "attendance")),
]
nav_columns = st.columns(len(navigation))
for column, (label, destination, protected, is_active) in zip(nav_columns, navigation):
    with column:
        if st.button(
            label,
            key=f"bottom_nav_{destination}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            navigate_to(destination, requires_admin=protected)
