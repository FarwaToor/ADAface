#!/usr/bin/env python3

import redis
from sqlite_utils import create_table_if_not_exists, store_person, retrieve_all_persons

# Initialize Redis client
r = redis.Redis(host='localhost', port=6379, db=0)

def migrate_redis_to_sqlite():
    """
    Migrate person data from Redis to SQLite.
    """
    # Ensure SQLite table exists
    create_table_if_not_exists()

    # Retrieve all keys from Redis
    redis_keys = r.keys()

    for key in redis_keys:
        key_type = r.type(key).decode('utf-8')
        if key_type == 'hash':
            person_id = key.decode('utf-8').split('_')[1]  # Extract person ID from key
            data = r.hgetall(key)
            person_name = data.get(b'person_name').decode('utf-8')  # Retrieve person's name from Redis

            # Store person data in SQLite
            store_person(person_id, person_name)
            print(f"Migrated person_id '{person_id}' with name '{person_name}' to SQLite.")

    print("Migration completed.")

if __name__ == "__main__":
    migrate_redis_to_sqlite()

