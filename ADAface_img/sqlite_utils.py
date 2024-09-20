#!/usr/bin/env python3

import sqlite3

# SQLite database file
sqlite_db_file = 'person_data.db'

def create_table_if_not_exists():
    """
    Create the table for storing person data if it does not exist.
    """
    conn = sqlite3.connect(sqlite_db_file)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS person_data (
            person_id INTEGER PRIMARY KEY,
            person_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_person(person_id, person_name):
    """
    Store person data in SQLite.
    """
    conn = sqlite3.connect(sqlite_db_file)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO person_data (person_id, person_name)
        VALUES (?, ?)
    ''', (person_id, person_name))
    conn.commit()
    conn.close()

def retrieve_person(person_id):
    """
    Retrieve person data by person_id from SQLite.
    """
    conn = sqlite3.connect(sqlite_db_file)
    c = conn.cursor()
    c.execute('''
        SELECT person_name FROM person_data WHERE person_id = ?
    ''', (person_id,))
    row = c.fetchone()
    conn.close()
    if row:
        person_name = row[0]
        return person_name
    return None

def retrieve_all_persons():
    """
    Retrieve all person data from SQLite.
    """
    conn = sqlite3.connect(sqlite_db_file)
    c = conn.cursor()
    c.execute('SELECT person_id, person_name FROM person_data')
    rows = c.fetchall()
    conn.close()
    print(rows)
    return rows
    
def delete_person(person_id):
    """
    Delete person record from SQLite.
    """
    conn = sqlite3.connect(sqlite_db_file)
    c = conn.cursor()
    c.execute('''
        DELETE FROM person_data WHERE person_id = ?
    ''', (person_id,))
    conn.commit()
    conn.close()
    print(f"Deleted person with ID {person_id}")

# Example usage
if __name__ == "__main__":
    create_table_if_not_exists()
    print("Table creation or verification complete.")
    

