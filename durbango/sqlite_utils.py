def list_tables(cursor):
    """Convenience method to test state of class at different points in lifetime."""
    cursor.execute(''' SELECT name FROM sqlite_master  ''')
    return cursor.fetchall()


def run_query(sqlite_manager, query):
    with sqlite_manager.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        items = cursor.fetchall()
    return items
