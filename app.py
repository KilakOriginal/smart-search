from flask import Flask, abort, render_template, request, jsonify, send_from_directory
from logic.search import search as get_search_results, load_dictionary, load_document_lengths, precision_at_k, map_k, get_relevant_documents, DEFAULT_OUTPUT_DIR
from logic.utils import generate_title_with_ollama, setup_logging, time_it, get_image_paths
import logging
import argparse
from typing import Dict, List, Tuple, Union, Set
from pathlib import Path

DOCUMENTS_DIR = DEFAULT_OUTPUT_DIR.parent / "documents"
IMAGE_DIR = DEFAULT_OUTPUT_DIR.parent / "images"

DICTIONARY_ITEMS: Union[List[Tuple[str, int]], None] = None
DOCUMENT_LENGTHS: Union[Dict[int, int], None] = None
TOTAL_DOCUMENTS: int = 0
AVERAGE_DOCUMENT_LENGTH: float = None
DOCUMENT_IDS: Set[int] = None

POSTINGS_FILE_PATH = DEFAULT_OUTPUT_DIR / "postings"
DICTIONARY_FILE_PATH = DEFAULT_OUTPUT_DIR / "postings_dictionary"
DOCUMENT_LENGTHS_FILE_PATH = DEFAULT_OUTPUT_DIR / "document_lengths"

STOPWORDS = ["the", "is", "a", "an", "and", "or", "of"]

PREVIEW_MAX_LENGTH = 200

def load_index_data():
    global DICTIONARY_ITEMS, DOCUMENT_LENGTHS, TOTAL_DOCUMENTS, AVERAGE_DOCUMENT_LENGTH, DOCUMENT_IDS
    logging.info("Loading search index data...")

    dictionary = load_dictionary(DICTIONARY_FILE_PATH)
    if dictionary:
        logging.info(f"Main dictionary loaded with {len(dictionary)} terms.")
        DICTIONARY_ITEMS = list(dictionary.items())
    else:
        logging.error(f"Failed to load main dictionary from {DICTIONARY_FILE_PATH}. Search will be impaired.")
        DICTIONARY_ITEMS = []

    DOCUMENT_LENGTHS = load_document_lengths(DOCUMENT_LENGTHS_FILE_PATH)
    if DOCUMENT_LENGTHS:
        TOTAL_DOCUMENTS = len(DOCUMENT_LENGTHS)
        logging.info(f"Document lengths loaded for {TOTAL_DOCUMENTS} documents.")
    else:
        logging.error(f"Failed to load document lengths from {DOCUMENT_LENGTHS_FILE_PATH}.")
        DOCUMENT_LENGTHS = {}
        TOTAL_DOCUMENTS = 0
    AVERAGE_DOCUMENT_LENGTH = sum(DOCUMENT_LENGTHS.values()) / TOTAL_DOCUMENTS if TOTAL_DOCUMENTS > 0 else 0
    logging.info(f"Average document length calculated: {AVERAGE_DOCUMENT_LENGTH:.2f} characters.")

    DOCUMENT_IDS = DOCUMENT_LENGTHS.keys() if DOCUMENT_LENGTHS else set()

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


@app.route('/get_title/<int:doc_id>')
def get_title(doc_id):
    try:
        file_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        if not file_path.is_relative_to(DOCUMENTS_DIR):
            logging.warning(f"Attempt to access file outside designated directory: {file_path}")
            abort(403)

        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            title = generate_title_with_ollama(content)
            return jsonify({"title": title, "doc_id": doc_id})
        else:
            logging.warning(f"Title requested for non-existent document: {doc_id}.txt at {file_path}")
            return jsonify({"title": f"Document {doc_id}"}), 404
    except Exception as e:
        logging.error(f"Error generating title for doc_id {doc_id}: {e}")
        return jsonify({"title": f"Document {doc_id}"}), 500


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

@app.route('/get_images')
def get_images():
    try:
        image_paths = get_image_paths(IMAGE_DIR)
        if not image_paths:
            logging.warning("No images found in the specified directory.")
            return jsonify({"images": []}), 404  # Not found

        logging.debug(f"Found {len(image_paths)} images in {IMAGE_DIR}")
        logging.debug(f"Image paths: {image_paths}")

        images = [{"path": f"/images/{path.name}", "name": path.name} for path in image_paths]
        return jsonify({"images": images})    
    except Exception as e:
        logging.error(f"Error fetching images: {e}")
        return jsonify({"error": "Error loading images."}), 500 # Internal server error

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from the IMAGE_DIR directory."""
    try:
        return send_from_directory(IMAGE_DIR, filename)
    except FileNotFoundError:
        abort(404)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    logging.debug("Search endpoint called with query: %s", request.args.get('q', ''))
    query = request.args.get('q', '')
    results = time_it(get_search_results, query, DICTIONARY_ITEMS, DOCUMENT_LENGTHS, AVERAGE_DOCUMENT_LENGTH, TOTAL_DOCUMENTS, DOCUMENT_IDS, POSTINGS_FILE_PATH, STOPWORDS)

    logging.info(f"Precision at 10: {precision_at_k(results, get_relevant_documents(results), 10):.2f}")
    logging.info(f"Mean Average Precision at 10: {map_k(results, get_relevant_documents(results), 10):.2f}")

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
