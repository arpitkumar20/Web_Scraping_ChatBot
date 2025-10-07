import os
import datetime
from psycopg2 import OperationalError
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from app.helper.utils import COMMON
from app.core.logging import get_logger


# ------------------ Configure Logging ------------------
logger = get_logger(__name__)

DATABASE_HOST = os.getenv('DATABASE_HOST')
DATABASE_PORT = os.getenv('DATABASE_PORT')
DATABASE_USER = os.getenv('DATABASE_USER')
DATABASE_PASS = os.getenv('DATABASE_PASS')
DATABASE_NAME = os.getenv('DATABASE_NAME')
TABLE_NAME = os.getenv('TABLE_NAME')

DATABASE_CONFIG = {
    "host": DATABASE_HOST,
    "port": DATABASE_PORT,
    "user": DATABASE_USER,
    "password": DATABASE_PASS,
    "dbname": DATABASE_NAME
}

class PostgreSQL:
    def __init__(self, database_host=DATABASE_HOST, database_port=DATABASE_PORT,
                 database_user=DATABASE_USER, database_pass=DATABASE_PASS,
                 database_name=DATABASE_NAME):
        try:
            if not hasattr(self.__class__, 'pg_connection_pool') or self.__class__.pg_connection_pool is None:
                self.__class__.pg_connection_pool = SimpleConnectionPool(
                    1, 20,
                    host=database_host,
                    port=database_port,
                    user=database_user,
                    password=database_pass,
                    dbname=database_name
                )
            
            self.__class__.pg_connection = self.__class__.pg_connection_pool.getconn()
            self.__class__.pg_cursor = self.__class__.pg_connection.cursor(cursor_factory=RealDictCursor)
            logger.info('✅ PostgreSQL connection established successfully.')

        except OperationalError as error:
            logger.error(f'❌ PostgreSQL connection error: {error}')
            raise

    def __del__(self):
        try:
            if hasattr(self.__class__, 'pg_cursor') and self.__class__.pg_cursor:
                self.__class__.pg_cursor.close()
                logger.info('🔒 PostgreSQL cursor closed.')

            if hasattr(self.__class__, 'pg_connection') and self.__class__.pg_connection:
                self.__class__.pg_connection_pool.putconn(self.__class__.pg_connection)
                logger.info('🔒 PostgreSQL connection returned to pool.')

        except Exception as error:
            logger.error(f'Error during __del__: {error}')

    def connection_test(self):
        try:
            self.__class__.pg_cursor.execute('SELECT version()')
            server_version = self.__class__.pg_cursor.fetchone()
            logger.info(f'PostgreSQL connection test successful: {server_version}')
            return {'message': 'Connection successful', 'pgsql_version': server_version}, 200
        except Exception as error:
            logger.error(f'Connection test error: {error}')
            return {'error': str(error)}, 401

    def list_schema(self):
        try:
            self.__class__.pg_cursor.execute('SELECT nspname FROM pg_catalog.pg_namespace;')
            schema_list = self.__class__.pg_cursor.fetchall()
            logger.info('Schema list fetched successfully.')
            return {'schemas': [schema['nspname'] for schema in schema_list]}, 200
        except Exception as error:
            logger.error(f'Schema fetch error: {error}')
            return {'error': str(error)}, 401

    def list_tables(self, schema_name):
        try:
            query = """
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = %s AND table_type='BASE TABLE'
            """
            self.__class__.pg_cursor.execute(query, (schema_name,))
            table_list = self.__class__.pg_cursor.fetchall()
            logger.info(f'Tables in schema "{schema_name}" fetched successfully.')
            return {'tables': [table['table_name'] for table in table_list]}, 200
        except Exception as error:
            logger.error(f'Table list error: {error}')
            return {'error': str(error)}, 401

    def list_columns(self, schema_name, table_name):
        try:
            query = """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
            """
            self.__class__.pg_cursor.execute(query, (schema_name, table_name))
            column_list = self.__class__.pg_cursor.fetchall()
            logger.info(f'Columns for table "{schema_name}.{table_name}" fetched successfully.')
            return {'columns': [col['column_name'] for col in column_list]}, 200
        except Exception as error:
            logger.error(f'Column list error: {error}')
            return {'error': str(error)}, 401

    def scan_row(self, schema_name=None, table_and_columns=None, primary_column=None, query=None, offset=0, limit=100):
        try:
            results = []
            if schema_name and table_and_columns:
                for record in table_and_columns:
                    columns_str = ','.join(record['columns']) if 'columns' in record else '*'
                    run_query = f'SELECT {columns_str} FROM {schema_name}.{record.get("table_name")} LIMIT {limit} OFFSET {offset}'
                    self.__class__.pg_cursor.execute(run_query)
                    row_list = self.__class__.pg_cursor.fetchall()
                    for row in row_list:
                        record_dict = {}
                        for col, value in row.items():
                            result = COMMON.stringify(value)
                            record_dict[col] = str(result) if col == primary_column else result
                        results.append(record_dict)
                logger.info(f'Scanned rows from schema "{schema_name}".')
                return results
            elif query:
                self.__class__.pg_cursor.execute(query)
                row_list = self.__class__.pg_cursor.fetchall()
                for row in row_list:
                    record_dict = {}
                    for col, value in row.items():
                        result = COMMON.stringify(value)
                        record_dict[col] = str(result) if col == primary_column else result
                    results.append(record_dict)
                logger.info('Custom query scan executed successfully.')
                return results
            else:
                logger.warning('No schema/table or query provided for scan_row.')
                return []
        except Exception as error:
            logger.error(f'Scan row error: {error}')
            return {'error': str(error)}, 401

    # def insert_message_data(self, data):
    #     try:
    #         insert_query = f"""
    #             INSERT INTO {TABLE_NAME} (
    #                 status, whatsappMessageId, localMessageId, text, type, time, message_status,
    #                 statusString, isOwner, ticketId, assignedId, sourceType, isDeleted, translationStatus,
    #                 message_id, tenantId, created, conversationId, channelType, airesponse
    #             ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    #         """

    #         message_time = data['message']['time']
    #         if isinstance(message_time, str):
    #             message_time = int(message_time)

    #         is_owner = data['message']['isOwner']
    #         if isinstance(is_owner, str):
    #             is_owner = is_owner.lower() in ['true', '1']

    #         is_deleted = data['message']['isDeleted']
    #         if isinstance(is_deleted, str):
    #             is_deleted = is_deleted.lower() in ['true', '1']

    #         created_time = data['message']['created']
    #         if isinstance(created_time, str):
    #             created_time = datetime.datetime.fromisoformat(created_time.replace('Z', '+00:00')[:26])

    #         values = (
    #             data['status'],
    #             data['message']['whatsappMessageId'],
    #             data['message']['localMessageId'],
    #             data['text'],
    #             data['message']['type'],
    #             message_time,
    #             data['message']['status'],
    #             data['message']['statusString'],
    #             is_owner,
    #             data['message']['ticketId'],
    #             data['message']['assignedId'],
    #             data['message']['sourceType'],
    #             is_deleted,
    #             data['message']['translationStatus'],
    #             data['message']['id'],
    #             data['message']['tenantId'],
    #             created_time,
    #             data['message']['conversationId'],
    #             data['message']['channelType'],
    #             data['airesponse']
    #         )

    #         self.__class__.pg_cursor.execute(insert_query, values)
    #         self.__class__.pg_connection.commit()
    #         logger.info("Record inserted successfully into PostgreSQL.")
    #         return {"message": "Record inserted successfully."}, 200

    #     except Exception as error:
    #         logger.error(f'Insert operation failed: {error}')
    #         return {"message": "Insert operation failed.", "description": str(error)}, 500

    # Safe int conversion helper
    def safe_int(self, value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None  # or raise error if you want strict validation

    def insert_rds_message_data(self, data):
        try:
            insert_query = f"""
                INSERT INTO {TABLE_NAME} (
                    status, whatsappMessageId, localMessageId, text, type, time, message_status,
                    statusString, isOwner, ticketId, assignedId, sourceType, isDeleted, translationStatus,
                    message_id, tenantId, created, conversationId, channelType, airesponse
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # --- Convert message_time ---
            message_time = data['message']['time']
            if isinstance(message_time, str) and message_time.isdigit():
                message_time = datetime.datetime.fromtimestamp(int(message_time))
            elif isinstance(message_time, int):
                message_time = datetime.datetime.fromtimestamp(message_time)
            elif isinstance(message_time, str):
                message_time = datetime.datetime.fromisoformat(message_time.replace('Z', '+00:00'))

            # --- Boolean conversion ---
            is_owner = data['message']['isOwner']
            if isinstance(is_owner, str):
                is_owner = is_owner.lower() in ['true', '1']

            is_deleted = data['message']['isDeleted']
            if isinstance(is_deleted, str):
                is_deleted = is_deleted.lower() in ['true', '1']

            # # --- Convert created time ---
            # created_time = data['message']['created']
            # if isinstance(created_time, str):
            #     created_time = datetime.datetime.fromisoformat(created_time.replace('Z', '+00:00')[:26])
            
            created_time = data['message']['created']
            if isinstance(created_time, str):
                try:
                    if created_time.endswith('+'):
                        created_time = created_time.rstrip('+') + '+00:00'
                    created_time = datetime.datetime.fromisoformat(created_time)
                except ValueError:
                    # fallback parser
                    from dateutil import parser
                    created_time = parser.isoparse(created_time)

            values = (
                data['status'],
                data['message']['whatsappMessageId'],
                data['message']['localMessageId'],
                data['text'],
                data['message']['type'],
                message_time,
                data['message']['status'],
                data['message']['statusString'],
                is_owner,
                self.safe_int(data['message']['ticketId']),
                self.safe_int(data['message']['assignedId']),
                data['message']['sourceType'],
                is_deleted,
                data['message']['translationStatus'],
                data['message']['id'],
                self.safe_int(data['message']['tenantId']),
                created_time,
                self.safe_int(data['message']['conversationId']),
                data['message']['channelType'],
                data['airesponse']
            )

            # ✅ Create a fresh cursor for this insert
            with self.__class__.pg_connection.cursor() as cur:
                cur.execute(insert_query, values)
                self.__class__.pg_connection.commit()

            logger.info("Record inserted successfully into PostgreSQL.")
            return {"message": "Record inserted successfully."}, 200

        except Exception as error:
            logger.error(f'Insert operation failed: {error}')
            return {"message": "Insert operation failed.", "description": str(error)}, 500


    def insert_web_scraping_status(self, data: dict):
        """
        Insert job result into job_results table.
        Missing fields will be inserted as NULL.
        """

        try:
            insert_query = f"""
                INSERT INTO web_scraping_status (
                    job_id, message, namespace, url, status
                ) VALUES (%s, %s, %s, %s, %s)
            """

            # Extract fields safely, defaulting to None if missing
            job_id = data.get("job_id")
            message = data.get("message")
            namespace = data.get("namespace") or None
            url = data.get("url") or None
            status = data.get("status")

            values = (job_id, message, namespace, url, status)

            with self.pg_connection.cursor() as cur:
                cur.execute(insert_query, values)
                self.pg_connection.commit()

            logger.info("✅ Record inserted successfully into PostgreSQL.")
            return {"message": "Record inserted successfully."}, 200

        except Exception as error:
            logger.error(f'❌ Insert operation failed: {error}')
            return {
                "message": "Insert operation failed.",
                "description": str(error)
            }, 500
        
    def update_web_scraping_status(self, data: dict):
        """
        Update job result in web_scraping_status table.
        Updates the row matching the given job_id.
        """
        try:
            update_query = """
                UPDATE web_scraping_status
                SET message = %s,
                    namespace = %s,
                    url = %s,
                    status = %s
                WHERE job_id = %s
            """

            # Extract fields safely
            job_id = data.get("job_id")
            message = data.get("message")
            namespace = data.get("namespace") or None
            url = data.get("url") or None
            status = data.get("status")

            values = (message, namespace, url, status, job_id)

            with self.pg_connection.cursor() as cur:
                cur.execute(update_query, values)
                self.pg_connection.commit()

            logger.info(f"✅ Record updated successfully for job_id={job_id}.")
            return {"message": "Record updated successfully."}, 200

        except Exception as error:
            logger.error(f'❌ Update operation failed: {error}')
            return {
                "message": "Update operation failed.",
                "description": str(error)
            }, 500

