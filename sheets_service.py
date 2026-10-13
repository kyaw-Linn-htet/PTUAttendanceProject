import os

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

from database.db import get_db


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HEADERS = [
    "Local attendance ID",
    "Student ID",
    "Student name",
    "Class/session",
    "Date",
    "Check-in time",
    "Status",
    "Confidence",
]


def _required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def _quoted_sheet_range(worksheet, cell_range):
    # Escape apostrophes in worksheet names for A1 notation.
    safe_name = worksheet.replace("'", "''")
    return f"'{safe_name}'!{cell_range}"


def export_attendance_session(session_id):
    """Append new attendance rows for a session; return the number appended."""
    spreadsheet_id = _required_env("GOOGLE_SHEETS_SPREADSHEET_ID")
    credentials_path = _required_env("GOOGLE_APPLICATION_CREDENTIALS")
    worksheet = os.environ.get("GOOGLE_SHEETS_WORKSHEET", "Attendance").strip()

    if not worksheet:
        worksheet = "Attendance"

    credentials = Credentials.from_service_account_file(
        credentials_path,
        scopes=SCOPES,
    )
    sheets = build(
        "sheets",
        "v4",
        credentials=credentials,
        cache_discovery=False,
    ).spreadsheets()

    db = get_db()
    records = db.execute(
        """
        SELECT
            a.id AS local_attendance_id,
            s.student_id,
            s.name,
            x.session_name,
            a.session_date,
            a.check_in_timestamp,
            a.status,
            a.recognition_confidence
        FROM attendance AS a
        JOIN students AS s ON s.id = a.student_id
        JOIN attendance_sessions AS x ON x.id = a.session_id
        WHERE a.session_id = ?
        ORDER BY a.id
        """,
        (session_id,),
    ).fetchall()

    sheet_range = _quoted_sheet_range(worksheet, "A:H")
    id_range = _quoted_sheet_range(worksheet, "A:A")

    existing = (
        sheets.values()
        .get(spreadsheetId=spreadsheet_id, range=id_range)
        .execute()
        .get("values", [])
    )

    # Column A stores the local attendance record ID for deduplication.
    existing_ids = {
        str(row[0])
        for row in existing[1:]
        if row and row[0] not in ("", None)
    }

    if not existing:
        sheets.values().update(
            spreadsheetId=spreadsheet_id,
            range=_quoted_sheet_range(worksheet, "A1"),
            valueInputOption="RAW",
            body={"values": [HEADERS]},
        ).execute()

    new_rows = [
        [
            record["local_attendance_id"],
            record["student_id"],
            record["name"],
            record["session_name"],
            record["session_date"],
            record["check_in_timestamp"],
            record["status"],
            record["recognition_confidence"],
        ]
        for record in records
        if str(record["local_attendance_id"]) not in existing_ids
    ]

    if new_rows:
        sheets.values().append(
            spreadsheetId=spreadsheet_id,
            range=sheet_range,
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": new_rows},
        ).execute()

    return len(new_rows)
