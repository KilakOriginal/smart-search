from flask import Flask, render_template, request, jsonify
from logic.search import prefix_search, direct_search, load_dictionary, DEFAULT_OUTPUT_DIR
from logic.utils import setup_logging, time_it
import logging
import argparse
from typing import Dict, List, Tuple, Union
from pathlib import Path

DICTIONARY_ITEMS: Union[List[Tuple[str, int]], None] = None
#SKIP_LIST: Union[Dict[str, int], None] = None

POSTINGS_FILE_PATH = DEFAULT_OUTPUT_DIR / "postings"
DICTIONARY_FILE_PATH = DEFAULT_OUTPUT_DIR / "postings_dictionary"
#SKIP_LIST_FILE_PATH = DICTIONARY_FILE_PATH.with_suffix('.skip')

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
