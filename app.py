import os
import re
import json
import sqlite3
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, g
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("HERBLOOM_DB_PATH", BASE_DIR / "herbloom_users.sqlite3"))

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("HERBLOOM_SECRET_KEY", "dev-only-change-this-secret-key"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("HERBLOOM_HTTPS", "0") == "1",
    MAX_CONTENT_LENGTH=1 * 1024 * 1024,
)

# Load the existing trained PCOS model. Keep the model and feature order unchanged.
model = joblib.load(BASE_DIR / "pcos_model.pkl")
features = [
    " Age (yrs)",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)"
]


def get_db():
    if "db" not in g:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS period_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            periods_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )""")


def logged_in_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    row = get_db().execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        session.clear()
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def require_login(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = logged_in_user()
        if user is None:
            return jsonify({"error": "Please log in to access account diary data."}), 401
        g.current_user = user
        return fn(*args, **kwargs)
    return wrapper


@app.route("/")
def home():
    return render_template("index.html")


@app.post("/api/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if not name or len(name) > 80:
        return jsonify({"error": "Please enter your name (up to 80 characters)."}), 400
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254:
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 8 or len(password) > 128:
        return jsonify({"error": "Your password must be between 8 and 128 characters."}), 400
    db = get_db()
    try:
        cursor = db.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, generate_password_hash(password))
        )
        user_id = cursor.lastrowid
        db.execute("INSERT INTO period_entries (user_id, periods_json) VALUES (?, '[]')", (user_id,))
        db.commit()
    except sqlite3.IntegrityError:
        db.rollback()
        return jsonify({"error": "An account with that email already exists. Please log in instead."}), 409
    session.clear()
    session["user_id"] = user_id
    session.permanent = True
    return jsonify({"user": {"id": user_id, "name": name, "email": email}}), 201


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    row = get_db().execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Email or password is incorrect."}), 401
    session.clear()
    session["user_id"] = row["id"]
    session.permanent = True
    return jsonify({"user": {"id": row["id"], "name": row["name"], "email": row["email"]}})


@app.get("/api/auth/me")
def me():
    return jsonify({"user": logged_in_user()})


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/periods", methods=["GET", "PUT"])
@require_login
def account_periods():
    db = get_db()
    user_id = g.current_user["id"]
    if request.method == "GET":
        row = db.execute("SELECT periods_json FROM period_entries WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            db.execute("INSERT INTO period_entries (user_id, periods_json) VALUES (?, '[]')", (user_id,))
            db.commit()
            return jsonify([])
        try:
            return jsonify(json.loads(row["periods_json"]))
        except (TypeError, json.JSONDecodeError):
            return jsonify([])

    data = request.get_json(silent=True) or {}
    periods = data.get("periods")
    if not isinstance(periods, list) or len(periods) > 2000:
        return jsonify({"error": "The diary data format is not valid."}), 400
    # Store only expected fields and cap text lengths to keep data predictable.
    cleaned = []
    for item in periods:
        if not isinstance(item, dict):
            continue
        start, end = str(item.get("start", "")), str(item.get("end", ""))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end):
            continue
        symptoms = item.get("symptoms", [])
        if not isinstance(symptoms, list):
            symptoms = []
        cleaned.append({
            "id": str(item.get("id", ""))[:100],
            "start": start,
            "end": end,
            "flow": str(item.get("flow", "")).strip()[:40],
            "symptoms": [str(s)[:80] for s in symptoms[:30]],
            "notes": str(item.get("notes", ""))[:2000],
        })
    db.execute("""INSERT INTO period_entries (user_id, periods_json, updated_at)
                  VALUES (?, ?, CURRENT_TIMESTAMP)
                  ON CONFLICT(user_id) DO UPDATE SET
                    periods_json = excluded.periods_json,
                    updated_at = CURRENT_TIMESTAMP""", (user_id, json.dumps(cleaned, ensure_ascii=False)))
    db.commit()
    return jsonify({"ok": True, "count": len(cleaned)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        values = [
            data["age"], data["cycle"], data["duration"], data["weightGain"],
            data["hairGrowth"], data["skin"], data["hairLoss"], data["pimples"],
            data["fastFood"], data["exercise"]
        ]
        input_data = pd.DataFrame([values], columns=features)
        prediction = model.predict(input_data)[0]
        if str(prediction).strip().lower() in ["1", "1.0", "yes", "y"]:
            message = ("The model estimates a higher likelihood of PCOS. This is not a diagnosis. "
                       "Please consult a qualified healthcare professional for assessment.")
        else:
            message = ("The model estimates a lower likelihood of PCOS. This does not rule out PCOS "
                       "or other health conditions. Consult a healthcare professional if you have concerns.")
        return jsonify({"message": message})
    except Exception:
        return jsonify({"error": "Could not calculate the estimate. Please check your inputs and model setup."}), 400


init_db()

if __name__ == "__main__":
    # debug=False avoids exposing the interactive debugger to other devices.
    app.run(debug=False)