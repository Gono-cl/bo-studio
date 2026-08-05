import sqlite3
import json
from io import StringIO
import pandas as pd
from datetime import datetime
import os
from core.utils.app_paths import get_db_path

DB_NAME = str(get_db_path())
os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)


def _safe_json_load(text, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _safe_read_results_json(text):
    if not text:
        return pd.DataFrame()
    try:
        return pd.read_json(StringIO(text), orient="records")
    except ValueError:
        return pd.DataFrame()


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            name TEXT,
            timestamp TEXT,
            notes TEXT,
            variables_json TEXT,
            results_json TEXT,
            best_result_json TEXT,
            settings_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_experiment(user_email, name, notes, variables, df_results, best_result, settings):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Serialize best_result as JSON (works for both dict and list)
    if best_result is not None:
        best_result_json = json.dumps(best_result)
    else:
        best_result_json = None

    cursor.execute("""
        INSERT INTO experiments (
            user_email, name, timestamp, notes, variables_json, results_json, best_result_json, settings_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_email,
        name,
        timestamp,
        notes,
        json.dumps(variables),
        df_results.to_json(orient="records"),
        best_result_json,
        json.dumps(settings)
    ))

    conn.commit()
    conn.close()


def list_experiments(user_email):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, timestamp FROM experiments WHERE user_email = ? ORDER BY id DESC", (user_email,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def load_experiment(exp_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, timestamp, notes, variables_json, results_json, best_result_json, settings_json
        FROM experiments
        WHERE id = ?
    """, (exp_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        name, timestamp, notes, var_json, res_json, best_json, settings_json = row
        return {
            "name": name,
            "timestamp": timestamp,
            "notes": notes,
            "variables": _safe_json_load(var_json, []),
            "df_results": _safe_read_results_json(res_json),
            "best_result": _safe_json_load(best_json, None),
            "settings": _safe_json_load(settings_json, {}),
        }
    else:
        return None


def delete_experiments(exp_ids):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.executemany("DELETE FROM experiments WHERE id = ?", [(i,) for i in exp_ids])
    conn.commit()
    conn.close()

