import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'knuckle.db')
print(f"DB path: {db_path}")
print(f"Exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    print(f"User count: {c.fetchone()[0]}")
    c.execute("SELECT uid, name FROM users")
    users = c.fetchall()
    print(f"Users: {users}")
    conn.close()
