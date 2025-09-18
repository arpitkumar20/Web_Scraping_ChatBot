import logging
from flask import Blueprint, request, jsonify
from app.models.mysql_db import MySQL
from app.models.postgresql_db import PostgreSQL
from app.services.store_vectordb_embeddings import process_and_store_data


database_bp = Blueprint("database", __name__)

# Initialize MySQL connection
@database_bp.route("/connection-test", methods=["POST"])
def init_connection():
    global mysql_instance
    try:
        data = request.json
        if data.get('database_type') == 'mysql':
            mysql_instance = MySQL(
                database_host=data.get('database_host'),
                database_port=data.get('database_port'),
                database_user=data.get('database_user'),
                database_pass=data.get('database_pass'),
                database_name=data.get('database_name')
            )
            return mysql_instance.connection_test()
        elif data.get('database_type') == 'postgresql':
            postgres_instance = PostgreSQL(
                database_host=data.get('database_host'),
                database_port=data.get('database_port'),
                database_user=data.get('database_user'),
                database_pass=data.get('database_pass'),
                database_name=data.get('database_name')
            )
            return postgres_instance.connection_test()
        else:
            return jsonify({"message": "Connector type doesn't exists."}), 200
    except Exception as e:
        logging.error(f"Error initializing Database connection: {e}")
        return jsonify({"error": str(e)}), 500


@database_bp.route("/database-info", methods=["POST"])
def database_info():
    try:
        data = request.json
        connector_type = data.get('connector_type')  # Required: 'mysql' or 'postgresql'
        schema_name = data.get('schema_name')
        table_name = data.get('table_name')
        columns = data.get('columns')        # List of column names (optional)

        if not connector_type:
            return jsonify({"error": "connector_type is required"}), 400

        # Initialize DB instance dynamically based on connector_type
        if connector_type == 'mysql':
            db_instance = MySQL()  # Initialize with required connection settings
        elif connector_type == 'postgresql':
            db_instance = PostgreSQL()  # Initialize with required connection settings
        else:
            return jsonify({"error": f"Unsupported connector_type '{connector_type}'"}), 400

        # Step 1: List Schemas
        if not schema_name:
            result = db_instance.list_schema()
            return jsonify({"operation": "list_schema", "result": result}), 200

        # Step 2: List Tables in Schema
        if schema_name and not table_name:
            result = db_instance.list_tables(schema_name)
            return jsonify({
                "operation": "list_tables",
                "schema_name": schema_name,
                "result": result
            }), 200

        # Step 3: List Columns in Table
        if schema_name and table_name and not columns:
            result = db_instance.list_columns(schema_name, table_name)
            return jsonify({
                "operation": "list_columns",
                "schema_name": schema_name,
                "table_name": table_name,
                "result": result
            }), 200

        return jsonify({"error": "Invalid parameter combination"}), 400

    except Exception as e:
        logging.error(f"Database operation error: {e}")
        return jsonify({"error": str(e)}), 500

@database_bp.route("/scan-rows", methods=["POST"])
def scan_rows():
    try:
        data = request.json
        connector_type = data.get('connector_type')  # Required: 'mysql' or 'postgresql'
        schema_name = data.get('schema_name')
        table_and_columns = data.get('table_and_columns')  # List of dicts: [{'table_name': 'users', 'columns': ['id', 'name']}]
        query = data.get('query')
        primary_column = data.get('primary_column')
        offset = data.get('offset', 0)
        limit = data.get('limit', 100)

        if not connector_type:
            return jsonify({"error": "connector_type is required"}), 400

        if not (schema_name and table_and_columns) and not query:
            return jsonify({"error": "Either (schema_name and table_and_columns) or query is required"}), 400

        # Initialize DB instance dynamically
        if connector_type == 'mysql':
            db_instance = MySQL()
        elif connector_type == 'postgresql':
            db_instance = PostgreSQL()
        else:
            return jsonify({"error": f"Unsupported connector_type '{connector_type}'"}), 400

        # Execute scan row operation
        result = db_instance.scan_row(
            schema_name=schema_name,
            table_and_columns=table_and_columns,
            query=query,
            primary_column=primary_column,
            offset=offset,
            limit=limit
        )

        return jsonify({
            "operation": "scan_rows",
            "schema_name": schema_name,
            "table_and_columns": table_and_columns,
            "primary_column": primary_column,
            "offset": offset,
            "limit": limit,
            "result": result
        }), 200

    except Exception as e:
        logging.error(f"Error scanning rows: {e}")
        return jsonify({"error": str(e)}), 500


@database_bp.route("/store-embeddings", methods=["POST"])
def store_embeddings():
    try:
        data = request.json
        connector_type = data.get('connector_type')  # Required: 'mysql' or 'postgresql'
        schema_name = data.get('schema_name')
        table_and_columns = data.get('table_and_columns')  # List of dicts: [{'table_name': 'users', 'columns': ['id', 'name']}]
        query = data.get('query')
        primary_column = data.get('primary_column')
        offset = data.get('offset', 0)
        limit = data.get('limit', 100)

        if not connector_type:
            return jsonify({"error": "connector_type is required"}), 400

        if not (schema_name and table_and_columns) and not query:
            return jsonify({"error": "Either (schema_name and table_and_columns) or query is required"}), 400

        # Initialize DB instance dynamically
        if connector_type == 'mysql':
            db_instance = MySQL()
        elif connector_type == 'postgresql':
            db_instance = PostgreSQL()
        else:
            return jsonify({"error": f"Unsupported connector_type '{connector_type}'"}), 400

        # Execute scan row operation
        result = db_instance.scan_row(
            schema_name=schema_name,
            table_and_columns=table_and_columns,
            query=query,
            primary_column=primary_column,
            offset=offset,
            limit=limit
        )

        store_embedding_response = process_and_store_data(result_list=result)
        print("Record sucessfully stored into vectorDb ",store_embedding_response)

        return jsonify({
            "operation": "scan_rows",
            "schema_name": schema_name,
            "table_and_columns": table_and_columns,
            "primary_column": primary_column,
            "offset": offset,
            "limit": limit,
            "result": result
        }), 200

    except Exception as e:
        logging.error(f"Error scanning rows: {e}")
        return jsonify({"error": str(e)}), 500