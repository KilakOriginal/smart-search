from flask import Flask, abort, render_template, request, jsonify
from logic.search import prefix_search, direct_search, load_dictionary, DEFAULT_OUTPUT_DIR
from logic.utils import setup_logging, time_it
import logging
import argparse
from typing import Dict, List, Tuple, Union
from pathlib import Path

DOCUMENTS_DIR = DEFAULT_OUTPUT_DIR.parent / "documents"

DICTIONARY_ITEMS: Union[List[Tuple[str, int]], None] = None
#SKIP_LIST: Union[Dict[str, int], None] = None

POSTINGS_FILE_PATH = DEFAULT_OUTPUT_DIR / "postings"
DICTIONARY_FILE_PATH = DEFAULT_OUTPUT_DIR / "postings_dictionary"
#SKIP_LIST_FILE_PATH = DICTIONARY_FILE_PATH.with_suffix('.skip')

PREVIEW_MAX_LENGTH = 150

def load_index_data():
    global DICTIONARY_ITEMS #, SKIP_LIST
    logging.info("Loading search index data...")

    dictionary = load_dictionary(DICTIONARY_FILE_PATH)
    if dictionary:
        logging.info(f"Main dictionary loaded with {len(dictionary)} terms.")
        DICTIONARY_ITEMS = list(dictionary.items())
    else:
        logging.error(f"Failed to load main dictionary from {DICTIONARY_FILE_PATH}. Search will be impaired.")
        DICTIONARY_ITEMS = []

    #SKIP_LIST = load_dictionary(SKIP_LIST_FILE_PATH)
    #if SKIP_LIST:
    #    logging.info(f"Skip list loaded with {len(SKIP_LIST)} terms.")
    #else:
    #    logging.warning(f"Failed to load skip list from {SKIP_LIST_FILE_PATH}.")

    logging.info("Search index data loading complete.")

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Text search application')

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except critical errors."
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug output."
    )

    return parser.parse_args()


# Initialise logging
args = parse_args()
setup_logging(args)

# Initialise Flask application
app = Flask(__name__)
load_index_data()

@app.route('/get_preview/<int:doc_id>')
def get_preview(doc_id):
    try:
        length_str = request.args.get('length')
        length = PREVIEW_MAX_LENGTH
        if length_str and length_str.isdigit():
            length = min(int(length_str), length)

        file_path = DOCUMENTS_DIR / f"{doc_id}.txt"

        if not file_path.is_relative_to(DOCUMENTS_DIR):
            logging.warning(f"Attempt to access file outside designated directory: {file_path}")
            abort(403)  # Forbidden

        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(length)
                if len(content) == length and f.read(1):
                    content += "..." # Indicate truncation
                return jsonify({"preview": content, "doc_id": doc_id})
        else:
            logging.warning(f"Preview requested for non-existent document: {doc_id}.txt at {file_path}")
            return jsonify({"preview": "No content found."}), 404 # Not found
    except Exception as e:
        logging.error(f"Error fetching preview for doc_id {doc_id}: {e}")
        return jsonify({"preview": "Error loading preview."}), 500 # Internal server error

@app.route('/document/<int:doc_id>')
def get_document_page(doc_id):
    try:
        file_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        if not file_path.is_relative_to(DOCUMENTS_DIR):
            logging.warning(f"Attempt to access file outside designated directory: {file_path}")
            abort(403)

        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return render_template('document.html', doc_id=doc_id, content=content)
        else:
            logging.warning(f"Full document requested for non-existent document: {doc_id}.txt at {file_path}")
            abort(404) # Page not found
    except Exception as e:
        logging.error(f"Error serving document page for doc_id {doc_id}: {e}")
        abort(500) # Internal server error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    logging.debug("Search endpoint called with query: %s", request.args.get('q', ''))
    query = request.args.get('q', '')
    if query.endswith('*'):
        results = time_it(prefix_search, query[:-1], DICTIONARY_ITEMS)
    else:
        results = time_it(direct_search, query, DICTIONARY_ITEMS)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
