from flask import Flask, render_template, request, jsonify
from logic.search import prefix_search

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    results = prefix_search(query)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)