import sqlite3
import pandas as pd

def init_db():
    """Requirement 2.6: Create the database and table if they don't exist."""
    conn = sqlite3.connect('bookings.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            service TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_booking(name, email, phone, service):
    """Save a new booking to the SQLite database."""
    conn = sqlite3.connect('bookings.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO appointments (name, email, phone, service)
        VALUES (?, ?, ?, ?)
    ''', (name, email, phone, service))
    conn.commit()
    conn.close()

def get_all_bookings():
    """Retrieve all bookings for the Admin Dashboard."""
    conn = sqlite3.connect('bookings.db')
    # Using pandas makes it easy to display in Streamlit
    df = pd.read_sql_query("SELECT * FROM appointments ORDER BY timestamp DESC", conn)
    conn.close()
    return df