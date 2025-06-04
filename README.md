# Smart Search
## About
*Smart Search* is a simple search engine that supports prefix, phrase and boolean text queries. The results page displays all relevant documents in descending order according to BM25+. There are a maximum of 10 results per page. Each result consists of a title and a preview. You can also view the full documents by clicking on their respective titles.

## Setup
1. Create a virtual environment (optional, but strongly recommended)
2. Install the requirements using `pip install -r requirements.txt`
3. Add a postings file, dictionary and document length file to `static/index/` and export the documents to `static/documents/` or change the paths in `app.py` or `logic/search.py` to point to your own files.
4. Install ollama and pull the `llama3` model using `ollama pull llama3`

## Usage
1. Start the ollama server using `ollama serve`
2. Start the flask server using `python app.py`
3. Navigate to the server address in your webbrowser (`http://127.0.0.1:5000` by default)
4. Search the corpus of documents!

### Flags
|Argument|Description|Default|
|---|---|---|
|`-v`, `--verbose`|Enable verbose output.|False|
|`-q`, `--quiet`|Suppress all output.|False|
|`-d`, `--debug`|Enable debug output.|False|

## Syntax
- **Phrase queries**: Simply type the phrase into the search box and press enter or click on the magnifying glass icon.
- **Prefix queries**: Type in a term followed by a `*` and press enter or click on the magnifying glass icon.
- **Boolean queris**: Use `!` (exclamation point) for a logical not, `^` (circumflex) for a logical and, `|` (pipe) for a logical or, `~` (tilde) for a logical exclusive or, `/<distance>` (solidus + number) for a near operation and use `()` (parantheses) for prioritisation. Additionally, a `\` (reverse solidus) can be used to escape these special characters (including the reverse solidus itself). Surround parts of the input with `"` (double quotes) to mark them as phrases.

## Future Development
- Add prefix queries
- Add search suggestions

## Contributing
If you want to contribute to this project, feel free to open an issue or a pull request. All contributions are welcome!

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.
