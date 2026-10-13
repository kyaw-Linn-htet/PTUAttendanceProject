from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import os
import json
import base64
import numpy as np
from database.db import init_db, get_db
from modules.face_utils import encode_face, compare_faces, detect_face_in_image
import io
from PIL import Image
import pickle
import atexit
import re
from datetime import datetime, timedelta, timezone
from threading import RLock
from camera_service import camera_service

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'ptu_face_recognition_secret_2026')
app.config['UPLOAD_FOLDER'] = 'static/uploads/faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'PTUG4')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'PTU2026')

LOW_CONFIDENCE_THRESHOLD = 50.0
MEDIUM_CONFIDENCE_THRESHOLD = 60.0

MAJORS = [
    'Civil Engineering',
    'Electronic Engineering',
    'Mechanical Engineering',
    'Electrical Power Engineering',
    'Computer Engineering and Information Technology',
]

SEMESTERS = ['Semester', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

with app.app_context():
    init_db()

# ─── User Panel ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('user/index.html')

@app.route('/search', methods=['POST'])
def search_student():
    student_id = request.form.get('student_id', '').strip()
    if not student_id:
        return render_template('user/index.html', error='Please enter a Student ID.')
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE student_id = ?', (student_id,)).fetchone()
    if student:
        return render_template('user/result.html', student=dict(student))
    else:
        return render_template('user/index.html', error=f'No student found with ID: {student_id}')

@app.route('/detect', methods=['GET', 'POST'])
def detect_face():
    if request.method == 'POST':
        result = None
        error = None
        image_data = None

        if 'face_image' in request.files and request.files['face_image'].filename:
            file = request.files['face_image']
            img_bytes = file.read()
            image_data = 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode()
            result, error = recognize_from_bytes(img_bytes)
        elif request.form.get('captured_image'):
            b64 = request.form['captured_image'].split(',')[1]
            img_bytes = base64.b64decode(b64)
            image_data = request.form['captured_image']
            result, error = recognize_from_bytes(img_bytes)
        else:
            error = 'No image provided.'

        return render_template('user/detect.html', result=result, error=error, image_data=image_data)
    return render_template('user/detect.html')

def recognize_from_bytes(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)
        encoding = encode_face(img_array)
        if encoding is None:
            return None, 'No face detected in the image. Please try again with a clear face photo.'
        db = get_db()
        students = db.execute('SELECT * FROM students WHERE face_encodings IS NOT NULL').fetchall()
        best_match = None
        best_distance = float('inf')
        for student in students:
            try:
                stored = pickle.loads(student['face_encodings'])
                for enc in stored:
                    dist = compare_faces(enc, encoding)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = dict(student)
            except Exception:
                continue
        if best_match:
            raw_confidence = (1 - best_distance) * 100
            confidence = round(raw_confidence, 1)
            if raw_confidence < LOW_CONFIDENCE_THRESHOLD:
                return None, 'The Student has not been recorded in the Database'
            if raw_confidence < MEDIUM_CONFIDENCE_THRESHOLD:
                return None, 'Please Make More Clear Photo'
            best_match['confidence'] = confidence
            return best_match, None
        return None, 'Same image not found in Database.'
    except Exception as e:
        return None, f'Error processing image: {str(e)}'

# ─── Admin Panel ──────────────────────────────────────────────────────────────

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        flash('Invalid credentials. Please try again.', 'danger')
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/admin')
@admin_required
def admin_dashboard():
    db = get_db()
    total = db.execute('SELECT COUNT(*) as c FROM students').fetchone()['c']
    trained = db.execute('SELECT COUNT(*) as c FROM students WHERE face_encodings IS NOT NULL').fetchone()['c']
    recent = db.execute('SELECT * FROM students ORDER BY created_at DESC LIMIT 5').fetchall()
    return render_template('admin/dashboard.html', total=total, trained=trained, recent=[dict(r) for r in recent])

@app.route('/admin/students')
@admin_required
def admin_students():
    db = get_db()
    students = db.execute('SELECT * FROM students ORDER BY created_at DESC').fetchall()
    return render_template('admin/students.html', students=[dict(s) for s in students], majors=MAJORS, semesters=SEMESTERS)

@app.route('/admin/students/add', methods=['GET', 'POST'])
@admin_required
def add_student():
    if request.method == 'POST':
        name = request.form['name'].strip()
        student_id = request.form['student_id'].strip()
        major = request.form['major']
        semester = request.form['semester']
        roll_number = request.form['roll_number'].strip()

        db = get_db()
        existing = db.execute('SELECT id FROM students WHERE student_id = ?', (student_id,)).fetchone()
        if existing:
            flash(f'Student ID {student_id} already exists.', 'danger')
            return render_template('admin/add_student.html', majors=MAJORS, semesters=SEMESTERS)

        db.execute(
            'INSERT INTO students (name, student_id, major, semester, roll_number) VALUES (?, ?, ?, ?, ?)',
            (name, student_id, major, semester, roll_number)
        )
        db.commit()
        flash(f'Student {name} added successfully!', 'success')
        return redirect(url_for('admin_students'))
    return render_template('admin/add_student.html', majors=MAJORS, semesters=SEMESTERS)

@app.route('/admin/students/edit/<int:sid>', methods=['GET', 'POST'])
@admin_required
def edit_student(sid):
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE id = ?', (sid,)).fetchone()
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('admin_students'))
    if request.method == 'POST':
        name = request.form['name'].strip()
        student_id = request.form['student_id'].strip()
        major = request.form['major']
        semester = request.form['semester']
        roll_number = request.form['roll_number'].strip()
        db.execute(
            'UPDATE students SET name=?, student_id=?, major=?, semester=?, roll_number=? WHERE id=?',
            (name, student_id, major, semester, roll_number, sid)
        )
        db.commit()
        flash('Student updated successfully!', 'success')
        return redirect(url_for('admin_students'))
    return render_template('admin/edit_student.html', student=dict(student), majors=MAJORS, semesters=SEMESTERS)

@app.route('/admin/students/delete/<int:sid>', methods=['POST'])
@admin_required
def delete_student(sid):
    db = get_db()
    db.execute('DELETE FROM students WHERE id = ?', (sid,))
    db.execute('DELETE FROM face_images WHERE student_id = ?', (sid,))
    db.commit()
    flash('Student deleted.', 'success')
    return redirect(url_for('admin_students'))

@app.route('/admin/train/<int:sid>', methods=['GET', 'POST'])
@admin_required
def train_student(sid):
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE id = ?', (sid,)).fetchone()
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('admin_students'))
    images = db.execute('SELECT * FROM face_images WHERE student_id = ?', (sid,)).fetchall()
    if request.method == 'POST':
        files = request.files.getlist('face_images')
        saved = 0
        for f in files:
            if f and f.filename:
                img_bytes = f.read()
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    img_array = np.array(img)
                    enc = encode_face(img_array)
                    if enc is None:
                        continue
                    img_b64 = base64.b64encode(img_bytes).decode()
                    db.execute(
                        'INSERT INTO face_images (student_id, image_data) VALUES (?, ?)',
                        (sid, img_b64)
                    )
                    saved += 1
                except Exception:
                    continue
        db.commit()
        if saved > 0:
            retrain_student(sid, db)
            flash(f'{saved} image(s) uploaded and trained successfully!', 'success')
        else:
            flash('No valid face images detected. Please upload clear face photos.', 'danger')
        return redirect(url_for('train_student', sid=sid))
    return render_template('admin/train.html', student=dict(student), images=[dict(i) for i in images])

@app.route('/admin/train/<int:sid>/delete_image/<int:img_id>', methods=['POST'])
@admin_required
def delete_face_image(sid, img_id):
    db = get_db()
    db.execute('DELETE FROM face_images WHERE id = ? AND student_id = ?', (img_id, sid))
    db.commit()
    retrain_student(sid, db)
    flash('Image deleted.', 'success')
    return redirect(url_for('train_student', sid=sid))

@app.route('/admin/retrain_all', methods=['POST'])
@admin_required
def retrain_all():
    db = get_db()
    students = db.execute('SELECT id FROM students').fetchall()
    count = 0
    for s in students:
        if retrain_student(s['id'], db):
            count += 1
    flash(f'Retrained {count} students successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

def retrain_student(sid, db):
    images = db.execute('SELECT image_data FROM face_images WHERE student_id = ?', (sid,)).fetchall()
    if len(images) < 1:
        db.execute('UPDATE students SET face_encodings = NULL WHERE id = ?', (sid,))
        db.commit()
        return False
    encodings = []
    for img_row in images:
        try:
            img_bytes = base64.b64decode(img_row['image_data'])
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img)
            enc = encode_face(img_array)
            if enc is not None:
                encodings.append(enc)
        except Exception:
            continue
    if encodings:
        db.execute('UPDATE students SET face_encodings = ? WHERE id = ?', (pickle.dumps(encodings), sid))
        db.commit()
        return True
    return False


# Attendance capture is owned by one background camera thread; routes remain responsive.
_attendance_lock = RLock()
_active_session_id = None
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))
MYANMAR_TZ = timezone(timedelta(hours=6, minutes=30), name='MMT')


def myanmar_now():
    """Return the current Myanmar Standard Time as a SQLite-friendly value."""
    return datetime.now(MYANMAR_TZ).replace(tzinfo=None).isoformat(sep=' ', timespec='seconds')


def myanmar_today():
    return datetime.now(MYANMAR_TZ).date().isoformat()


def recognize_frame(frame):
    """Run the existing face encoder/comparator against this session's enrolled roster."""
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encoding = encode_face(rgb)
    if encoding is None:
        return {'message': 'No face detected.', 'student': None}
    with app.app_context():
        db = get_db()
        with _attendance_lock:
            session_id = _active_session_id
        if session_id is None:
            return {'message': 'No active attendance session.', 'student': None}
        rows = db.execute('''SELECT s.* FROM students s JOIN attendance_roster r
                             ON r.student_id=s.id WHERE r.session_id=? AND s.face_encodings IS NOT NULL''',
                          (session_id,)).fetchall()
        best, distance = None, float('inf')
        for row in rows:
            try:
                for known in pickle.loads(row['face_encodings']):
                    d = compare_faces(known, encoding)
                    if d < distance:
                        best, distance = dict(row), d
            except Exception:
                continue
        if best is None:
            return {'message': 'No registered student match.', 'student': None}
        raw_confidence = (1 - distance) * 100
        confidence = round(raw_confidence, 1)
        if raw_confidence < LOW_CONFIDENCE_THRESHOLD:
            return {'message': 'The Student has not been recorded in the Database', 'student': None,
                    'confidence': confidence}
        if raw_confidence < MEDIUM_CONFIDENCE_THRESHOLD:
            return {'message': 'Please Make More Clear Photo', 'student': None,
                    'confidence': confidence}
        try:
            db.execute('''INSERT OR IGNORE INTO attendance
                (student_id, session_id, session_date, check_in_timestamp, status, recognition_confidence)
                VALUES (?, ?, ?, ?, 'present', ?)''',
                       (best['id'], session_id, myanmar_today(), myanmar_now(), raw_confidence))
            db.commit()
        except Exception:
            db.rollback()
            return {'message': 'Attendance database write failed.', 'student': None,
                    'confidence': confidence}
        best['confidence'] = confidence
        return {'message': 'Recognized', 'student': best, 'confidence': confidence}


def attendance_payload(session_id):
    db = get_db()
    sess = db.execute('SELECT * FROM attendance_sessions WHERE id=?', (session_id,)).fetchone()
    if not sess:
        return {'session': None, 'present': [], 'absent': [], 'total': 0, 'count': 0}
    present = db.execute('''SELECT a.*, s.name, s.student_id AS external_id FROM attendance a
        JOIN students s ON s.id=a.student_id WHERE a.session_id=? ORDER BY a.check_in_timestamp''',
                         (session_id,)).fetchall()
    absent = []
    if sess['status'] == 'ended':
        absent = db.execute('''SELECT s.name, s.student_id AS external_id FROM attendance_roster r
            JOIN students s ON s.id=r.student_id LEFT JOIN attendance a
            ON a.student_id=r.student_id AND a.session_id=r.session_id
            WHERE r.session_id=? AND a.id IS NULL ORDER BY s.name''', (session_id,)).fetchall()
    total = db.execute('SELECT COUNT(*) FROM attendance_roster WHERE session_id=?', (session_id,)).fetchone()[0]
    return {'session': dict(sess), 'present': [dict(x) for x in present],
            'absent': [dict(x) for x in absent], 'total': total, 'count': len(present),
            'percentage': round(100 * len(present) / total, 1) if total else 0}


@app.route('/admin/attendance')
@admin_required
def attendance_dashboard():
    with _attendance_lock:
        active_id = _active_session_id
    latest = get_db().execute('SELECT id FROM attendance_sessions ORDER BY id DESC LIMIT 1').fetchone()
    selected = active_id or (latest['id'] if latest else None)
    state = attendance_payload(selected) if selected else None
    return render_template('attendance.html', state=state, majors=MAJORS,
                           semesters=SEMESTERS[1:], active=bool(active_id), camera_index=CAMERA_INDEX)


@app.route('/admin/attendance/start', methods=['POST'])
@admin_required
def attendance_start():
    global _active_session_id
    with _attendance_lock:
        if _active_session_id or camera_service.state()['running']:
            return jsonify(error='An attendance session is already active.'), 409
        major = request.form.get('major', '').strip()
        semester = request.form.get('semester', '').strip()
        name = request.form.get('session_name', '').strip()[:100]
        try:
            camera_index = int(request.form.get('camera_index', CAMERA_INDEX))
            if camera_index < 0 or camera_index > 20:
                raise ValueError()
        except ValueError:
            return jsonify(error='Camera index must be an integer from 0 to 20.'), 400
        db = get_db()
        roster = db.execute('SELECT id FROM students WHERE major=? AND semester=?', (major, semester)).fetchall()
        if not roster:
            return jsonify(error='No students are enrolled in that major and semester.'), 400
        cur = db.execute('''INSERT INTO attendance_sessions
            (session_name, major, semester, session_date, started_at, status)
            VALUES (?, ?, ?, ?, ?, 'active')''',
                         (name or f'{major} {semester}', major, semester, myanmar_today(), myanmar_now()))
        _active_session_id = cur.lastrowid
        db.executemany('INSERT INTO attendance_roster(session_id, student_id) VALUES (?, ?)',
                       [(_active_session_id, x['id']) for x in roster])
        db.commit()
        sid = _active_session_id
    try:
        camera_service.start(camera_index, recognize_frame)
    except Exception as exc:
        with _attendance_lock:
            _active_session_id = None
        db.execute("UPDATE attendance_sessions SET status='ended', ended_at=? WHERE id=?", (myanmar_now(), sid))
        db.commit()
        return jsonify(error=str(exc)), 503
    return jsonify(ok=True, session_id=sid)


@app.route('/admin/attendance/stop', methods=['POST'])
@admin_required
def attendance_stop():
    global _active_session_id
    camera_service.stop()
    with _attendance_lock:
        sid, _active_session_id = _active_session_id, None
    if sid:
        db = get_db()
        db.execute("UPDATE attendance_sessions SET status='ended', ended_at=? WHERE id=?", (myanmar_now(), sid))
        db.commit()
    return jsonify(ok=True)


@app.route('/admin/attendance/status')
@admin_required
def attendance_status():
    with _attendance_lock:
        sid = _active_session_id
    if sid is None:
        latest = get_db().execute('SELECT id FROM attendance_sessions ORDER BY id DESC LIMIT 1').fetchone()
        sid = latest['id'] if latest else None
    payload = attendance_payload(sid) if sid else None
    return jsonify({'running': camera_service.state()['running'], 'camera_error': camera_service.state()['error'],
                    'recognition': camera_service.state()['latest'], 'attendance': payload})


@app.route('/admin/attendance/video')
@admin_required
def attendance_video():
    from flask import Response
    def stream():
        import time
        while True:
            frame = camera_service.frame()
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.06)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/admin/attendance/export/<int:session_id>', methods=['POST'])
@admin_required
def attendance_export(session_id):
    try:
        from sheets_service import export_attendance_session

        exported = export_attendance_session(session_id)
        return jsonify(
            ok=True,
            exported=exported,
            message=f'Exported {exported} new attendance row(s).',
        )
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception('Google Sheets attendance export failed')
        return jsonify(error='Google Sheets export failed. Check credentials, sharing, worksheet, and network.'), 502


@app.route('/admin/attendance/review', methods=['POST'])
@admin_required
def attendance_review():
    """Teacher correction: mark a roster student present or remove their check-in."""
    sid, student_pk, action = request.form.get('session_id', ''), request.form.get('student_id', ''), request.form.get('action', '')
    if not sid.isdigit() or not student_pk.isdigit() or action not in ('present', 'absent'):
        return jsonify(error='Invalid attendance correction.'), 400
    db = get_db()
    allowed = db.execute('SELECT 1 FROM attendance_roster WHERE session_id=? AND student_id=?', (sid, student_pk)).fetchone()
    if not allowed:
        return jsonify(error='Student is not enrolled in this session.'), 400
    if action == 'absent':
        db.execute('DELETE FROM attendance WHERE session_id=? AND student_id=?', (sid, student_pk))
    else:
        db.execute('''INSERT OR IGNORE INTO attendance
            (student_id, session_id, session_date, check_in_timestamp, status, recognition_confidence)
            SELECT ?, ?, session_date, ?, 'present', 100 FROM attendance_sessions WHERE id=?''',
                   (student_pk, sid, myanmar_now(), sid))
    db.commit()
    return jsonify(ok=True)


atexit.register(camera_service.stop)

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1', host='0.0.0.0', port=5000)
