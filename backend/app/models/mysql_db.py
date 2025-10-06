import os
import datetime
import logging
import mysql.connector
from mysql.connector import Error
from app.helper.utils import COMMON
from app.core.logging import get_logger

# ------------------ Configure Logging ------------------
logger = get_logger(__name__)

pool_name = 'mysqlpool'
pool_size = 20

DATABASE_HOST = os.getenv('DATABASE_HOST')
DATABASE_PORT = os.getenv('DATABASE_PORT')
DATABASE_USER = os.getenv('DATABASE_USER')
DATABASE_PASS = os.getenv('DATABASE_PASS')
DATABASE_NAME = os.getenv('DATABASE_NAME')
TABLE_NAME = os.getenv('TABLE_NAME')

class MySQL:
    def __init__(self, database_host=DATABASE_HOST, database_port=DATABASE_PORT, database_user=DATABASE_USER, database_pass=DATABASE_PASS, database_name=DATABASE_NAME):
        try:
            self.mysql_connection_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name=pool_name,
                pool_size=pool_size,
                pool_reset_session=True,
                host=database_host,
                port=database_port,
                user=database_user,
                password=database_pass,
                database=database_name
            )
            self.mysql_connection = self.mysql_connection_pool.get_connection()
            self.mysql_cursor = self.mysql_connection.cursor()
            logger.info('MySQL connection established successfully.')
        except Error as error:
            logger.error(f'MySQL Connection Error: {error}')
            raise

    def __del__(self):
        if hasattr(self, 'mysql_cursor') and self.mysql_cursor:
            self.mysql_cursor.close
            logger.info('MySQL cursor closed.')
        if hasattr(self, 'mysql_connection') and self.mysql_connection:
            self.mysql_connection.close()
            logger.info('MySQL connection closed.')

    def connection_test(self):
        try:
            self.mysql_cursor.execute('SELECT version()')
            server_version = self.mysql_cursor.fetchone()
            logger.info(f'MySQL connection test successful. Version: {server_version}')
            return {'message': 'Connection successful', 'mysql_version': server_version}
        except Error as error:
            logger.error(f'Connection test error: {error}')
            return {'error': str(error)}

    def list_schema(self):
        try:
            self.mysql_cursor.execute('SELECT DATABASE() FROM DUAL;')
            schema_list = self.mysql_cursor.fetchall()
            logger.info('Schema list fetched successfully.')
            return {'schemas': [schema[0] for schema in schema_list]}
        except Error as error:
            logger.error(f'Schema fetch error: {error}')
            return {'error': str(error)}

    def list_tables(self, schema_name):
        try:
            self.mysql_cursor.execute(
                'SELECT table_name FROM information_schema.tables WHERE table_schema = %s',
                (schema_name,)
            )
            table_list = self.mysql_cursor.fetchall()
            logger.info(f'Table list for schema "{schema_name}" fetched successfully.')
            return {'tables': [table[0] for table in table_list]}
        except Error as error:
            logger.error(f'Table list error: {error}')
            return {'error': str(error)}

    def list_columns(self, schema_name, table_name):
        try:
            query = f'SHOW COLUMNS FROM {schema_name}.{table_name}'
            self.mysql_cursor.execute(query)
            column_list = self.mysql_cursor.fetchall()
            logger.info(f'Column list for table "{schema_name}.{table_name}" fetched successfully.')
            return {'columns': [column[0] for column in column_list]}
        except Error as error:
            logger.error(f'Column list error: {error}')
            return {'error': str(error)}


    def scan_row(self, schema_name=None, table_and_columns=None, primary_column=None, query=None, offset=0, limit=100):
        try:
            cursor = self.mysql_connection.cursor(dictionary=True)
            results = []

            if schema_name and table_and_columns:
                for record in table_and_columns:
                    columns_str = ','.join(record['columns']) if 'columns' in record else '*'
                    run_query = f'SELECT {columns_str} FROM {schema_name}.{record.get("table_name")} LIMIT {limit} OFFSET {offset}'
                    cursor.execute(run_query)
                    row_list = cursor.fetchall()

                    for row in row_list:
                        record_dict = {}
                        for col, value in row.items():
                            result = COMMON.stringify(value)
                            record_dict[col] = str(result) if col == primary_column else result
                        results.append(record_dict)

                logger.info(f'Scanned rows from tables in schema "{schema_name}".')
                return results

            elif query:
                cursor.execute(query)
                row_list = cursor.fetchall()

                for row in row_list:
                    record_dict = {}
                    for col, value in row.items():
                        result = COMMON.stringify(value)
                        record_dict[col] = str(result) if col == primary_column else result
                    results.append(record_dict)

                logger.info('Custom query executed successfully for row scan.')
                return results

            else:
                logger.warning('No schema/table or query provided for scan_row.')
                return []

        except Error as error:
            logger.error(f'Scan row error: {error}')
            return {'error': str(error)}

    def insert_message_data(self, data):
        try:
            if self.mysql_cursor is None:
                raise Exception("MySQL connection not available.")

            insert_query = f"""
                INSERT INTO {TABLE_NAME} (
                    status, whatsappMessageId, localMessageId, text, type, time, message_status,
                    statusString, isOwner, ticketId, assignedId, sourceType, isDeleted, translationStatus,
                    message_id, tenantId, created, conversationId, channelType, airesponse
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Convert time and boolean fields correctly
            message_time = data['message']['time']
            if isinstance(message_time, str):
                message_time = int(message_time)

            is_owner = data['message']['isOwner']
            if isinstance(is_owner, str):
                is_owner = is_owner.lower() in ['true', '1']

            is_deleted = data['message']['isDeleted']
            if isinstance(is_deleted, str):
                is_deleted = is_deleted.lower() in ['true', '1']

            created_time = data['message']['created']
            if isinstance(created_time, str):
                created_time = datetime.datetime.fromisoformat(created_time.replace('Z', '+00:00')[:26])

            values = (
                data['status'],
                data['message']['whatsappMessageId'],
                data['message']['localMessageId'],
                data['text'],
                data['message']['type'],
                datetime.datetime.fromtimestamp(message_time).strftime('%Y-%m-%d %H:%M:%S'),
                data['message']['status'],
                data['message']['statusString'],
                is_owner,
                data['message']['ticketId'],
                data['message']['assignedId'],
                data['message']['sourceType'],
                is_deleted,
                data['message']['translationStatus'],
                data['message']['id'],
                data['message']['tenantId'],
                created_time,
                data['message']['conversationId'],
                data['message']['channelType'],
                data['airesponse']
            )

            self.mysql_cursor.execute(insert_query, values)
            self.mysql_connection.commit()
            logger.info("Record inserted successfully into MySQL.")
            return {"message": "Record inserted successfully."}, 200

        except Error as error:
            logger.error(f"MySQL insert operation failed: {error}")
            return {
                "message": "MySQL insert operation failed.",
                "description": str(error)
            }, 500