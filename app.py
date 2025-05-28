from flask import Flask, render_template, request, jsonify
from logic.search import prefix_search
import logging
import argparse

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

def setup_logging(args: argparse.Namespace) -> None:
    """
    Configure logging based on verbosity arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    elif args.quiet:
        logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


# Initialise logging
args = parse_args()
setup_logging(args)

# Initialise Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    logging.debug("Search endpoint called with query: %s", request.args.get('q', ''))
    query = request.args.get('q', '')
    results = prefix_search(query)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
