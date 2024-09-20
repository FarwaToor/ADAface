import sqlite3


def inspect_db():
    conn = sqlite3.connect('person_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM person_data')
    rows = cursor.fetchall()
    conn.close()
    return rows

print(inspect_db())
