# Smart Search
## About
*Smart Search* is a simple search engine that supports both prefix and boolean text queries.

## Usage
1. Create a virtual environment (optional, but strongly recommended)
2. Install the requirements using `pip install -r requirements.txt`
3. Add a postings file, dictionary and skip list file to `static/index/` or change the paths in `app.py` or `logic/search.py` to point to your own files.
4. Run `python app.py`
5. Navigate to the server address in your webbrowser (`http://127.0.0.1:5000` by default)
6. Search the corpus of documents!

## Future Development
- [ ] More visually appealing rendering of search results
- [ ] Add document previews
- [ ] Add pagination to search results
- [ ] Render documents in a user-friendly way (clickable links, etc.)
- [ ] Improve core search algorithm (specifically ranking of results)

## Contributing
If you want to contribute to this project, feel free to open an issue or a pull request. All contributions are welcome!

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.
