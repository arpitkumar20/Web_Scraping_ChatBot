import os
import datetime
from psycopg2 import OperationalError
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv()

DATABASE_HOST = os.getenv('DATABASE_HOST')
DATABASE_PORT = os.getenv('DATABASE_PORT')
DATABASE_USER = os.getenv('DATABASE_USER')
DATABASE_PASS = os.getenv('DATABASE_PASS')
DATABASE_NAME = os.getenv('DATABASE_NAME')
TABLE_NAME = os.getenv('TABLE_NAME')



# ✅ PostgreSQL connection parameters (pure psycopg2 format)
DATABASE_CONFIG = {
    "host": DATABASE_HOST,
    "port": DATABASE_PORT,
    "user": DATABASE_USER,
    "password": DATABASE_PASS,
    "dbname": DATABASE_NAME   # ✅ IMPORTANT: psycopg2 uses "dbname" not "database"
}

class PostgreSQL:
    pg_connection_pool = None
    pg_connection = None
    pg_cursor = None

    def __init__(self):
        try:
            # Initialize pool only once
            if self.__class__.pg_connection_pool is None:
                self.__class__.pg_connection_pool = SimpleConnectionPool(
                    1, 20,
                    **DATABASE_CONFIG
                )

            # Get connection from pool
            self.__class__.pg_connection = self.__class__.pg_connection_pool.getconn()
            self.__class__.pg_cursor = self.__class__.pg_connection.cursor()
            print("✅ PostgreSQL connection established.")

        except OperationalError as error:
            print("❌ PostgreSQL connection error:", error)

    def __del__(self):
        if self.__class__.pg_connection is not None:
            self.__class__.pg_cursor.close()
            self.__class__.pg_connection_pool.putconn(self.__class__.pg_connection)
            print("🔒 PostgreSQL connection closed and returned to pool.")

    def connection_test(self):
        try:
            if self.__class__.pg_cursor is None:
                raise Exception("PostgreSQL connection not available.")

            self.__class__.pg_cursor.execute("SELECT version()")
            server_version = self.__class__.pg_cursor.fetchone()
            return {
                "message": "PostgreSQL connection established.",
                "pgsql_version": server_version
            }, 200

        except Exception as error:
            print("❌ Connection test failed:", error)
            return {
                "message": "PostgreSQL connection error occurred.",
                "description": str(error)
            }, 401

    def list_schema(self):
        try:
            if self.__class__.pg_cursor is None:
                raise Exception("PostgreSQL connection not available.")

            self.__class__.pg_cursor.execute("SELECT nspname FROM pg_catalog.pg_namespace;")
            schema_list = self.__class__.pg_cursor.fetchall()

            return {
                "operation_type": "schema",
                "records": [schema[0] for schema in schema_list]
            }, 200

        except Exception as error:
            print("❌ Schema fetch failed:", error)
            return {
                "message": "PostgreSQL fetch schema error occurred.",
                "description": str(error)
            }, 401

    def fetch_all_rows(self, table_name):
        """
        Fetch all rows from a given table.
        :param table_name: str (name of the table)
        :return: dict with column names and rows
        """
        try:
            if self.__class__.pg_cursor is None:
                raise Exception("PostgreSQL connection not available.")

            # ✅ Get column names dynamically
            self.__class__.pg_cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            col_names = [desc[0] for desc in self.__class__.pg_cursor.description]

            # ✅ Fetch all rows
            self.__class__.pg_cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.__class__.pg_cursor.fetchall()

            return {
                "operation_type": "fetch_all",
                "table": table_name,
                "columns": col_names,
                "rows": rows
            }, 200

        except Exception as error:
            print(f"❌ Failed to fetch rows from {table_name}:", error)
            return {
                "message": f"PostgreSQL fetch rows error occurred for table {table_name}.",
                "description": str(error)
            }, 401


    def insert_message_data(self, data):
        """
        Insert a message record into the 'nissa' table.
        :param data: Dict containing 'status', 'ok', 'result', and 'message' sub-dict
        :return: Dict with status message
        """
        try:
            if self.__class__.pg_cursor is None:
                raise Exception("PostgreSQL connection not available.")

            insert_query = f"""
                INSERT INTO {TABLE_NAME} (
                    status, whatsappMessageId, localMessageId, text, type, time, message_status,
                    statusString, isOwner, ticketId, assignedId, sourceType, isDeleted, translationStatus,
                    message_id, tenantId, created, conversationId, channelType, airesponse
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

            self.__class__.pg_cursor.execute(insert_query, (
                data['status'],
                data['message']['whatsappMessageId'],
                data['message']['localMessageId'],
                data['text'],
                data['message']['type'],
                int(data['message']['time']),
                data['message']['status'],
                data['message']['statusString'],
                data['message']['isOwner'],
                data['message']['ticketId'],
                data['message']['assignedId'],
                data['message']['sourceType'],
                data['message']['isDeleted'],
                data['message']['translationStatus'],
                data['message']['id'],
                data['message']['tenantId'],
                datetime.datetime.fromisoformat(data['message']['created'].replace('Z', '+00:00')[:26]),
                data['message']['conversationId'],
                data['message']['channelType'],
                data['airesponse']
            ))

            self.__class__.pg_connection.commit()

            return {"message": "Record inserted successfully."}, 200

        except Exception as error:
            print("❌ Insert operation failed:", error)
            return {
                "message": "PostgreSQL insert operation failed.",
                "description": str(error)
            }, 500

# # ✅ Example usage
# if __name__ == "__main__":
#     db = PostgreSQL()

#     # Test connection
#     result, status = db.connection_test()
#     print(result)

    # # List schemas
    # schemas, status = db.list_schema()
    # print(schemas)

    # # Fetch all rows from a table
    # table_name = TABLE_NAME   # 🔄 Replace with actual table name
    # rows, status = db.fetch_all_rows(table_name)
    # print(rows)


    # sample_data = {
    #     'status': 'received',
    #     'message': {
    #         'whatsappMessageId': 'wamid.HBgMOTE4MjQwNjUxNTc0FQIAERgSODlGODYyMEZFQTY5NjkyRjc2AA==',
    #         'localMessageId': '9a538255-dfb8-46d1-aae7-22ae91c2f0c9',
    #         'text': "Is there something else I can help you with?  I'm happy to assist.",
    #         'type': 'text',
    #         'time': '1757528750',
    #         'status': 1,
    #         'statusString': 'SENT',
    #         'isOwner': True,
    #         'ticketId': '68c174cfd30bf0dfbc5ea53f',
    #         'assignedId': '689cd4696c529546aa769dcc',
    #         'sourceType': 0,
    #         'isDeleted': False,
    #         'translationStatus': 0,
    #         'id': '68c1c2aeca00333f30330665',
    #         'tenantId': '482313',
    #         'created': '2025-09-10T18:25:50.3790731Z',
    #         'conversationId': '68c174ade720d5e0ee9c60b6',
    #         'channelType': 0
    #     },
    #     'airesponse': "Hi How can i help you?"
    # }
    # result, status = db.insert_message_data(sample_data)
    # print(result)

