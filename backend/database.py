import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'knuckle.db')


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            uid TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            email TEXT,
            dob TEXT,
            gender TEXT,
            address TEXT,
            feature_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    
    # Create storage folders
    storage = os.path.join(os.path.dirname(__file__), 'storage')
    os.makedirs(os.path.join(storage, 'images'), exist_ok=True)
    os.makedirs(os.path.join(storage, 'models'), exist_ok=True)


def save_user(uid, name, phone, email, dob, gender, address, feature_path):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO users 
        (uid, name, phone, email, dob, gender, address, feature_path) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (uid, name, phone, email, dob, gender, address, feature_path))
    conn.commit()
    conn.close()


def get_user_by_uid(uid):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT uid, name, feature_path, phone, email FROM users WHERE uid = ?", (uid,))
    user = c.fetchone()
    conn.close()
    return user


def get_all_users():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT uid, name, feature_path FROM users")
    users = c.fetchall()
    conn.close()
    return users
