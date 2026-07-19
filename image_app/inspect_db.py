#!/usr/bin/env python3
"""Quick CLI to dump every row of the local person_data.db SQLite table."""
import sqlite3

from config import SQLITE_DB_PATH


def inspect_db():
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM person_data')
    rows = cursor.fetchall()
    conn.close()
    return rows


if __name__ == '__main__':
    print(inspect_db())
