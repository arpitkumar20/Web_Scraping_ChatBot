from app.core.logging import get_logger
from flask import Blueprint, request, jsonify
from app.services.file_content import process_file


extract_documents = Blueprint("documents", __name__)
logger = get_logger(__name__)


@extract_documents.route('/extract-content', methods=['POST'])
def file_content():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No selected file'}), 400

        # Call the reusable function
        result = process_file(file)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f'Error in predictSync: {e}')
        return jsonify({'error': str(e)}), 500
