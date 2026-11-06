
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "shieldml.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT
                    (datetime('now', 'localtime')),
                amount REAL NOT NULL,
                risk_score REAL NOT NULL,
                prediction TEXT NOT NULL
            )
        """)


def save_transaction(amount, risk_score, prediction):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO transactions (amount, risk_score, prediction)
            VALUES (?, ?, ?)
        """, (float(amount), float(risk_score), str(prediction)))


def get_transactions():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, created_at, amount, risk_score, prediction
            FROM transactions
            ORDER BY id DESC
        """).fetchall()
    return [dict(row) for row in rows]


def clear_transactions():
    with get_connection() as conn:
        conn.execute("DELETE FROM transactions")
