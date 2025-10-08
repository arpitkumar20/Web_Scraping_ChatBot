import os
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

DATABASE_CONFIG = {
    "host": DATABASE_HOST,
    "port": DATABASE_PORT,
    "user": DATABASE_USER,
    "password": DATABASE_PASS,
    "dbname": DATABASE_NAME
}

class PostgreSQLV2:
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
        """
        Fetch all schema names from the database.
        """
        try:
            self.__class__.pg_cursor.execute('SELECT nspname FROM pg_catalog.pg_namespace;')
            schema_list = self.__class__.pg_cursor.fetchall()
            logger.info('Schema list fetched successfully.')
            return [schema['nspname'] for schema in schema_list], 200
        except Exception as error:
            logger.error(f'Schema fetch error: {error}')
            return [], 401

    def list_tables(self):
        """
        Fetch tables only from the public schema dynamically.
        """
        try:
            query = """
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type='BASE TABLE'
            """
            self.__class__.pg_cursor.execute(query)
            table_list = self.__class__.pg_cursor.fetchall()
            logger.info('Tables fetched successfully from public schema.')
            return [table['table_name'] for table in table_list], 200
        except Exception as error:
            logger.error(f'Table list error: {error}')
            return [], 401

    def scan_row(self):
        """
        Fetch all rows from tables in the public schema dynamically.
        """
        try:
            result = {}
            tables, status = self.list_tables()
            if status != 200 or not tables:
                logger.warning('No tables found in public schema.')
                return {}

            schema_data = {}
            for table in tables:
                query = f'SELECT * FROM public.{table}'
                self.__class__.pg_cursor.execute(query)
                rows = self.__class__.pg_cursor.fetchall()
                schema_data[table] = [
                    {col: str(COMMON.stringify(value)) for col, value in row.items()} 
                    for row in rows
                ]

            if schema_data:
                result['public'] = schema_data

            logger.info('Fetched rows from public schema successfully.')
            return result

        except Exception as error:
            logger.error(f'Scan row error: {error}')
            return {'error': str(error)}, 401
