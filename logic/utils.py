from pathlib import Path
import time
import argparse
import logging
import heapq
import re
import tarfile
import shutil
import struct
import io
from typing import Union, List, Tuple, Dict, Set
import ollama
from typing import Callable, Any

WDIR = Path(__file__).resolve().parent.parent / "static"
DEFAULT_OUTPUT_DIR = WDIR / "index"

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Read and extract compressed files.")

    parser.add_argument(
        "-s", "--source",
        type=str,
        default=None,
        help="Path to the source archive (e.g., .tar.gz file)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to the output directory for extracted files (default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
            "-l", "--link-file",
            type=str,
            default=None,
            help="Path to the link file inside the archive."
    )
    parser.add_argument(
            "-c", "--content-file",
            type=str,
            default=None,
            help="Path to the content file inside the archive."
    )
    parser.add_argument(
        "--stopword-file",
        type=str,
        default=DEFAULT_OUTPUT_DIR / "stop_words",
        help="Path to the stop word file."
    )
    parser.add_argument(
        "-r", "--replace",
        action="store_true",
        help="Overwrite existing files in the output directory."
    )

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

def time_it(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[float, Any]:
    """
    Measures the execution time of a function.

    Args:
        func (Callable[..., Any]): The function to time.
        *args (Any): Positional arguments for the function.
        **kwargs (Any): Keyword arguments for the function.

    Returns:
        tuple[float, Any]: A tuple containing the elapsed time in ms 
                           and the result of the function call.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = (time.time() - start_time) * 1000
    return elapsed_time, result

def generate_title_with_ollama(document: str, model: str = "tinyllama") -> str:
    """
    Generates a title for the given document using an LLM via Ollama Python library.

    Args:
        document (str): The document text to generate a title for.
        model (str): The Ollama model to use (default: "llama3").

    Returns:
        str: The generated title as a string.
    """

    prompt = (
        "Generate a concise and relevant title for the following document. "
        "Only return the title string, nothing else.\n\n"
        f"{document}"
    )
    response = ollama.generate(model=model, prompt=prompt)
    logging.debug(f"Generated title response: {response['response'].strip()}")
    return response['response'].strip('"').strip("'").strip()

def extract_documents(source_path: Path, content_file_name: str, destination_directory: Path = DEFAULT_OUTPUT_DIR / "documents", max_documents: int = 0, replace: bool = False) -> int:
    """
    Extracts documents from a compressed archive file.

    Args:
        source_path (Path): Path to the source archive (e.g., .tar.gz file).
        content_file_name (str): Name of the content file inside the archive.
        destination_directory (Path): Directory to save extracted documents.
        max_documents (int): Maximum number of documents to extract (0 for no limit).

    Returns:
        int: 0 on success, 1 on failure.
    """
    logging.debug(f"Extracting documents from '{source_path}/{content_file_name}' to '{destination_directory}' with max_documents={max_documents}")

    if not source_path.is_file():
        logging.error(f"No such file '{source_path}'!")
        return 1

    destination_directory.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(source_path, mode='r:*') as archive:
            member = archive.getmember(content_file_name)

            with archive.extractfile(member) as file:
                for line_idx, line in enumerate(file):
                    if max_documents > 0 and line_idx >= max_documents:
                        break
                    try:
                        (document_id, document) = line.decode().strip().split("\t", 1)
                        document_id = document_id.strip()
                        document = document.strip()

                        output_file_path = destination_directory / f"{document_id}.txt"

                        if output_file_path.is_file() and not replace:
                            logging.warning(f"File '{output_file_path}' already exists but the overwrite flag is set to False.")
                            continue

                        with open(output_file_path, "w", encoding="utf-8") as output_file:
                            output_file.write(document)
                    except Exception as e:
                        logging.error(f"Error processing line {line_idx} ('{line.decode().strip()[:50]}...'): {e}")
                        continue
    except Exception as e:
        logging.error(f"Error opening archive '{source_path}': {e}")
        return 1

    return 0

def get_normalised_word_frequency(source_path: Path, content_file_name: str, max_documents: int = 0) -> Union[tuple[dict[str, tuple[float, int]], int], None]:
    """
    Calculates normalised term frequencies (sum of tf/document_length) and document frequencies 
    for words in a collection.

    Args:
        source_path (Path): Path to the source archive (e.g., .tar.gz file).
        content_file_name (str): Name of the content file inside the archive.
        max_documents (int): Maximum number of documents to process (0 for no limit).

    Returns:
        Union[tuple[dict[str, tuple[float, int]], int], None]: 
            A tuple containing:
            - A dictionary where keys are words (str) and values are tuples:
              (sum_normalised_tf (float), document_frequency (int)).
            - The total number of documents processed (int).
            Returns None if an error occurs (e.g., file not found).
    """
    logging.debug(f"Counting word frequency from '{source_path}/{content_file_name}' with max_documents={max_documents}")

    if not source_path.is_file():
      logging.error(f"No such file '{source_path}'!")
      return None

    word_frequency: dict[str, tuple[float, int]] = {}
    processed_document_count = 0

    try:
        with tarfile.open(source_path, mode='r:*') as archive:
            member = archive.getmember(content_file_name)

            with archive.extractfile(member) as file:
                for line_idx, line in enumerate(file):
                    if max_documents > 0 and processed_document_count >= max_documents:
                        break
                    try:
                        (_, document) = line.decode().strip().split("\t", 1)
                        word_document_frequency: dict[str, int] = {} # Term frequency within the current document
                        split_document = document.split()
                        document_length = len(split_document)
                        if document_length == 0:
                            continue

                        for word in split_document:
                            word = word.lower()
                            word_document_frequency[word] = word_document_frequency.get(word, 0) + 1

                        for word, count in word_document_frequency.items():
                            current_sum_tf, current_df = word_frequency.get(word, (0.0, 0))
                            word_frequency[word] = (current_sum_tf + count / document_length, current_df + 1)
                    except Exception as e:
                        logging.error(f"Error processing line {line_idx} ('{line.decode().strip()[:50]}...'): {e}")
                        continue
                    processed_document_count += 1
    except Exception as e:
        logging.error(f"Error opening archive '{source_path}': {e}")
        return None

    return (word_frequency, processed_document_count)

def build_stop_word_list(normalised_word_frequency: dict[str, tuple[float, int]], document_count: int) -> list[str]:
    """
    Builds a list of stop words based on document frequency.

    Args:
        normalised_word_frequency (dict[str, tuple[float, int]]): Dictionary with words as keys 
            and (sum_normalised_tf, document_frequency) as values.
        document_count (int): Total number of documents in the collection.

    Returns:
        list[str]: A list of stop words.
    """
    logging.debug(f"Building stop word list")
    if document_count == 0:
        return []

    # Eliminate words that appear in 85% of documents
    stop_words = []
    for word, (_, document_frequency) in normalised_word_frequency.items():
        if document_frequency / document_count >= 0.85: #or document_frequency == 1:
            stop_words.append(word)

    return stop_words

def save_stop_word_list(stop_words: list[str], output_path: Path, replace: bool = False) -> int:
    """
    Save the stop words to a file.

    Args:
        stop_words (list[str]): List of stop words.
        output_path (Path): Path to the output file.
        replace (bool): Whether to overwrite the file if it exists.
    Returns:
        int: 0 on success, 1 on failure.
    """
    if output_path.is_file() and not replace:
        logging.warning(f"File '{output_path}' already exists but the overwrite flag is set to False.")
        return 1

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for word in stop_words:
                f.write(f"{word}\n")
            return 0
    except Exception as e:
        logging.error(f"Error saving stop words to '{output_path}': {e}")
        return 1

def load_stop_word_list(file_path: Path) -> list[str]:
    """
    Load the stop words from a file.

    Args:
        file_path (Path): Path to the stop words file.

    Returns:
        list[str]: List of stop words. Returns an empty list if the file is not found or an error occurs.
    """
    if not file_path.is_file():
        logging.error(f"No such file '{file_path}'!")
        return []

    stop_words = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stop_words.append(line.strip())
    except Exception as e:
        logging.error(f"Error loading stop words from '{file_path}': {e}")
        return []

    return stop_words

def build_postings_and_dictionary(source_path: Path, content_file_name: str, stop_words: list[str],
                                      output_directory: Path = DEFAULT_OUTPUT_DIR,
                                      postings_file_name: str = "postings", replace: bool = False) -> int:
    """
    Builds the postings file and dictionary using a two-pass approach (SPIMI-like).
    First pass: Creates temporary blocks of inverted indexes.
    Second pass: Merges blocks into a final postings file and creates the dictionary.

    Args:
        source_path (Path): Path to the source archive.
        content_file_name (str): Name of the content file inside the archive.
        stop_words (list[str]): List of stop words to exclude.
        output_directory (Path): Directory to save output files.
        postings_file_name (str): Base name for the postings file and dictionary.
        replace (bool): Whether to overwrite existing output files.

    Returns:
        int: 0 on success, 1 on failure.
    """
    logging.debug(f"Building postings file from '{content_file_name}' in '{source_path}'")

    if not source_path.is_file():
        logging.error(f"No such file '{source_path}'!")
        return 1

    bytes_per_posting = 8
    max_block_memory_bytes = 250 * 1024 * 1024

    output_directory.mkdir(parents=True, exist_ok=True)
    output_file_path = output_directory / postings_file_name
    dictionary_file_path = output_directory / f"{postings_file_name}_dictionary"
    temp_dir = output_directory / "temp_postings"

    if (output_file_path.is_file() or dictionary_file_path.is_file()) and not replace:
        logging.warning("Output files already exist but the overwrite flag is set to False.")
        return 1

    temp_dir.mkdir(exist_ok=True)

    block_count = 0
    current_block_inverted_index = {} # {term: {document_id: [pos1, pos2, ...]}}
    current_block_memory_usage = 0 # in bytes

    try:
        with tarfile.open(source_path, mode='r:*') as archive:
            member = archive.getmember(content_file_name)

            with archive.extractfile(member) as file:
                word_pattern = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", re.UNICODE)

                for document_count, line in enumerate(file):
                    if document_count % 100000 == 0:
                        logging.info(f"Processed {document_count} documents")

                    try:
                        (document_id, document) = line.decode().strip().split("\t", 1)

                        # Tokenise and normalise
                        words_in_document = word_pattern.findall(document.lower())

                        # Temporary term frequency dictionary
                        document_term_data = {} # {term: [pos1, pos2, ...]}
                        for position, word in enumerate(words_in_document):
                            if word in stop_words:
                                continue

                            if word not in document_term_data:
                                document_term_data[word] = []
                            document_term_data[word].append(position)

                        # Update current block inverted index
                        for term, positions in document_term_data.items():
                            if term not in current_block_inverted_index:
                                current_block_inverted_index[term] = {}
                            current_block_inverted_index[term][document_id] = (len(positions), positions) # (term_frequency, [positions])
                            current_block_memory_usage += bytes_per_posting

                        if current_block_memory_usage >= max_block_memory_bytes:
                            spill_block_to_disk(current_block_inverted_index, temp_dir, block_count)
                            block_count += 1
                            current_block_inverted_index.clear()
                            current_block_memory_usage = 0

                    except Exception as e:
                        logging.error(f"Error processing document: {e}")
                        continue

                # Spill the last block to disk
                if current_block_inverted_index:
                    spill_block_to_disk(current_block_inverted_index, temp_dir, block_count)
                    block_count += 1
                    current_block_inverted_index.clear()
                    current_block_memory_usage = 0

    except Exception as e:
        logging.error(f"Error building postings file: {e}")
        # Clean up temporary files if an error occurs
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return 1

    logging.info(f"Finished creating {block_count} temporary blocks. Starting merge.")

    # Merge temporary blocks and create final dictionary
    try:
        merge_blocks_and_build_dictionary(temp_dir, output_file_path, dictionary_file_path, block_count)
    except Exception as e:
        logging.error(f"Error during merging blocks: {e}")
        return 1
    finally:
        if temp_dir.exists():
            logging.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    logging.info(f"Postings file saved to '{output_file_path}'")
    return 0

def spill_block_to_disk(block_inverted_index: dict[str, dict[str, tuple[int, list[int]]]], temp_dir: Path, block_id: int) -> None:
    """
    Sorts the terms in the current in-memory block and writes them to a temporary file.
    Each line in the temporary file has the format:
    term<tab>document_id1:tf1:pos11,pos12,...;document_id2:tf2:pos21,pos22,...

    Args:
        block_inverted_index (dict[str, dict[str, tuple[int, list[int]]]]): 
            The in-memory inverted index block. Structure: 
            {term: {document_id_str: (term_frequency, [positions_list])}}.
        temp_dir (Path): Directory to store temporary block files.
        block_id (int): Identifier for the current block.
    """
    temp_file_path = temp_dir / f"block_{block_id}"
    sorted_terms = sorted(block_inverted_index.keys())

    with open(temp_file_path, 'w', encoding='utf-8') as f:
        for term in sorted_terms:
            postings_list_str_parts = []
            sorted_document_ids = sorted(block_inverted_index[term].keys(), key=lambda x: int(x) if x.isdigit() else x)

            for document_id in sorted_document_ids:
                term_freq, positions = block_inverted_index[term][document_id]
                postings_list_str_parts.append(f"{document_id}:{term_freq}:{','.join(map(str, positions))}")

            f.write(f"{term}\t{';'.join(postings_list_str_parts)}\n")

def merge_blocks_and_build_dictionary(temp_dir: Path, final_postings_path: Path, final_dictionary_path: Path, num_blocks: int) -> None:
    """
    Merges all temporary sorted block files into a single final postings file
    and builds the term dictionary (term -> offset in postings file) simultaneously.

    The final postings file stores postings for each term contiguously. Each posting is:
    - Document ID (4 bytes, big-endian unsigned int)
    - Term Frequency in document (2 bytes, big-endian unsigned short)
    - Positions (list of 4-byte big-endian unsigned ints)

    The dictionary file stores: term<tab>offset_in_postings_file

    Args:
        temp_dir (Path): Directory containing the temporary block files.
        final_postings_path (Path): Path to write the final merged postings file.
        final_dictionary_path (Path): Path to write the final dictionary file.
        num_blocks (int): The number of temporary block files to merge.
    """
    # Open all temporary block files for reading
    temp_files = []
    for i in range(num_blocks):
        temp_files.append(open(temp_dir / f"block_{i}", 'r', encoding='utf-8'))

    # Each element in the heap will be (term, line_content, file_index)
    min_heap: list[tuple[str, str, int]] = []
    for i, f in enumerate(temp_files):
        line = f.readline()
        if line:
            try:
                term = line.split('\t')[0]
                heapq.heappush(min_heap, (term, line, i))
            except IndexError:
                logging.error(f"Malformed line in block {i}: {line.strip()}")
                continue
            except Exception as e:
                logging.error(f"Error reading line from block {i}: {e}")
                continue

    final_dictionary: dict[str, int] = {} # {term: postings_file_offset}
    current_postings_offset = 0

    with open(final_postings_path, 'wb') as f:
        while min_heap:
            current_term, line_content, file_idx = heapq.heappop(min_heap)

            next_line = temp_files[file_idx].readline()
            if next_line:
                try:
                    next_term = next_line.split('\t')[0]
                    heapq.heappush(min_heap, (next_term, next_line, file_idx))
                except IndexError:
                    logging.error(f"Malformed line in block {file_idx}: {next_line.strip()}")
                    continue
                except Exception as e:
                    logging.error(f"Error reading line from block {file_idx}: {e}")
                    continue

            lines_for_current_term = [(line_content, file_idx)]

            # Check for other occurrences of the same term in the heap
            while min_heap and min_heap[0][0] == current_term:
                next_line_content, next_file_idx = heapq.heappop(min_heap)[1:]
                lines_for_current_term.append((next_line_content, next_file_idx))

                next_next_line = temp_files[next_file_idx].readline()
                if next_next_line:
                    try:
                        next_next_term = next_next_line.split('\t')[0]
                        heapq.heappush(min_heap, (next_next_term, next_next_line, next_file_idx))
                    except IndexError:
                        logging.error(f"Malformed line in block {next_file_idx}: {next_next_line.strip()}")
                        continue
                    except Exception as e:
                        logging.error(f"Error reading line from block {next_file_idx}: {e}")
                        continue

            merged_postings_map = {} # {document_id: (term_freq, [positions])}

            for line, _ in lines_for_current_term:
                try:
                    _, postings_str = line.strip().split('\t', 1)
                except ValueError:
                    logging.warning(f"Malformed line during merge: {line.strip()}. Skipping.")
                    continue
                except Exception as e:
                    logging.error(f"Error processing line '{line}': {e}")
                    continue

                individual_postings = postings_str.split(';')
                for posting in individual_postings:
                    try:
                        document_id_string, term_frequency_string, position_string = posting.split(':', 2)
                        document_id = int(document_id_string)
                        term_frequency = int(term_frequency_string)
                        positions = list(map(int, position_string.split(',')))

                        # Assuming document_id is unique across all blocks
                        merged_postings_map[document_id] = (term_frequency, positions)
                    except ValueError:
                        logging.warning(f"Malformed posting: {posting}. Skipping.")
                        continue
                    except Exception as e:
                        logging.error(f"Error processing posting '{posting}': {e}")
                        continue

            # Construct the final postings list
            binary_postings_buffer = io.BytesIO()

            try:
                sorted_merged_document_ids = sorted(merged_postings_map.keys())
            except Exception as e:
                logging.warning(f"Error sorting merged document IDs: {e}")
                continue

            for document_id in sorted_merged_document_ids:
                term_frequency, positions = merged_postings_map[document_id]
                binary_postings_buffer.write(struct.pack('>I', document_id))
                binary_postings_buffer.write(struct.pack('>H', term_frequency))
                for position in positions:
                    binary_postings_buffer.write(struct.pack('>I', position))

            binary_postings_data = binary_postings_buffer.getvalue()
            f.write(binary_postings_data)

            postings_data_length = len(binary_postings_data)
            final_dictionary[current_term] = current_postings_offset
            current_postings_offset += postings_data_length

    for f in temp_files:
        f.close()

    # Write the final dictionary to disk
    with open(final_dictionary_path, 'w', encoding='utf-8') as f:
        for term in sorted(final_dictionary.keys()):
            offset = final_dictionary[term]
            f.write(f"{term}\t{offset}\n")

def build_skip_list(dictionary_path: Path, step_size: int = 1000) -> int:
    """
    Builds a skip list for the dictionary file and saves it.
    The skip list maps selected terms to their byte offsets in the dictionary file.

    Args:
        dictionary_path (Path): Path to the dictionary file (term<tab>offset format).
        step_size (int): Interval for creating skip list entries (e.g., every 1000th term).

    Returns:
        int: 0 on success, 1 on failure.
    """
    if not dictionary_path.is_file():
        logging.error(f"No such file '{dictionary_path}'!")
        return 1

    if step_size <= 0:
        logging.warning("step_size for skip list must be positive. Defaulting to no skip list creation.")
        return 1

    skip_list = {}
    line_number = 0

    try:
        with open(dictionary_path, 'rb') as f:
            while True:
                current_line_offset = f.tell()
                line_bytes = f.readline()

                if not line_bytes:
                    break

                if line_number % step_size == 0:
                    try:
                        line = line_bytes.decode('utf-8').strip()
                        skip_list[line.split('\t')[0]] = current_line_offset
                    except IndexError:
                        logging.warning(f"Malformed line in dictionary: {line.strip()}. Skipping.")
                    except Exception as e:
                        logging.error(f"Error processing line '{line}': {e}")

                line_number += 1
    except Exception as e:
        logging.error(f"Error loading dictionary file '{dictionary_path}': {e}")
        return 1

    # Save the skip list to a file
    skip_list_path = dictionary_path.with_suffix('.skip')

    try:
        with open(skip_list_path, 'w', encoding='utf-8') as f:
            for term, offset in skip_list.items():
                f.write(f"{term}\t{offset}\n")
        logging.info(f"Skip list saved to '{skip_list_path}'")
    except Exception as e:
        logging.error(f"Error saving skip list to '{skip_list_path}': {e}")
        return 1

    return 0

def calculate_and_save_document_lengths(source_path: Path, content_file_name: str,
                                        output_directory: Path = DEFAULT_OUTPUT_DIR,
                                        output_file_name: str = "document_lengths",
                                        replace: bool = False) -> int:
    """
    Calculates the number of words (length) for each document in the collection
    and saves this information to a binary file. Each entry in the file consists of:
    - Document ID (4-byte unsigned int, big-endian)
    - Document Length (4-byte unsigned int, big-endian)

    Args:
        source_path (Path): Path to the source archive (e.g., .tar.gz file).
        content_file_name (str): Name of the content file inside the archive.
        output_directory (Path): Directory to save the output binary file.
        output_file_name (str): Name for the output binary file.
        replace (bool): Whether to overwrite the output file if it exists.

    Returns:
        int: 0 on success, 1 on failure.
    """
    logging.debug(f"Calculating document lengths from '{source_path}/{content_file_name}' for binary output.")

    if not source_path.is_file():
        logging.error(f"No such file '{source_path}'!")
        return 1

    output_directory.mkdir(parents=True, exist_ok=True)
    output_file_path = output_directory / output_file_name

    if output_file_path.is_file() and not replace:
        logging.warning(f"File '{output_file_path}' already exists but the overwrite flag is set to False.")
        return 1

    document_id_length_pairs: list[tuple[int, int]] = []

    try:
        with tarfile.open(source_path, mode='r:*') as archive:
            member = archive.getmember(content_file_name)

            with archive.extractfile(member) as file:
                for line_idx, line_bytes in enumerate(file):
                    try:
                        line_str = line_bytes.decode().strip()
                        document_id_str, document_content = line_str.split("\t", 1)
                        
                        try:
                            document_id_int = int(document_id_str)
                        except ValueError:
                            logging.error(f"Could not convert document ID '{document_id_str}' to integer on line {line_idx}. Skipping.")
                            continue

                        document_length = len(document_content.split())
                        document_id_length_pairs.append((document_id_int, document_length))

                    except ValueError:
                        logging.error(f"Malformed line {line_idx} (expected 'document_id\\tdocument'): '{line_str[:100]}...'. Skipping.")
                        continue
                    except Exception as e:
                        logging.error(f"Error processing line {line_idx} ('{line_str[:100]}...'): {e}")
                        continue
    except tarfile.TarError as e:
        logging.error(f"TarError opening or reading archive '{source_path}': {e}")
        return 1
    except KeyError:
        logging.error(f"Content file '{content_file_name}' not found in archive '{source_path}'.")
        return 1
    except Exception as e:
        logging.error(f"Error reading from archive '{source_path}': {e}")
        return 1

    logging.debug(f"Collected {len(document_id_length_pairs)} document ID-length pairs.")

    # Save the document ID and lengths to a binary file
    try:
        with open(output_file_path, "wb") as f:
            for document_id, length in document_id_length_pairs:
                f.write(struct.pack('>I', document_id))
                f.write(struct.pack('>I', length))
        logging.info(f"Document lengths saved to binary file '{output_file_path}'")
        return 0
    except IOError as e:
        logging.error(f"IOError saving document lengths to '{output_file_path}': {e}")
        return 1
    except struct.error as e:
        logging.error(f"Struct packing error while saving document lengths: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error saving document lengths to '{output_file_path}': {e}")
        return 1

def load_dictionary(file_path: Path) -> Union[Dict[str, int], None]:
    """
    Loads a dictionary from a file.
    The file is expected to have lines in the format: term<tab>offset sorted lexicographically by term.

    Args:
        file_path (Path): Path to the dictionary file.

    Returns:
        Union[Dict[str, int], None]: A dictionary mapping terms to their integer offsets.
                                     Returns None if the file is not found or an error occurs.
    """
    if not file_path.is_file():
        logging.error(f"No such file '{file_path}'!")
        return None

    dictionary = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    term, offset_str = line.strip().split('\t')
                    dictionary[term] = int(offset_str)
                except ValueError:
                    logging.warning(f"Malformed line in dictionary: {line.strip()}. Skipping.")
                    continue
                except Exception as e:
                    logging.error(f"Error processing line '{line.strip()}': {e}")
                    continue
    except Exception as e:
        logging.error(f"Error loading dictionary file '{file_path}': {e}")
        return None

    return dictionary

def get_postings(posting_file_path: Path, offsets: List[Tuple[int, int]], terms: List[str]) -> Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None]:
    """
    Retrieves postings for a list of terms given their start and end offsets in the postings file.

    Each posting consists of (document_id, term_frequency, [positions]).

    Args:
        posting_file_path (Path): Path to the binary postings file.
        offsets (list[tuple[int, int]]): List of (start_offset, end_offset) tuples for each term's
                                         data in the postings file. end_offset = -1 means read to EOF
                                         or until next logical term block.
        terms (list[str]): List of terms corresponding to the offsets.

    Returns:
        Union[list[tuple[str, list[tuple[int, int, list[int]]]]], None]: 
            A list of tuples, where each tuple is (term, list_of_postings).
            Each posting in list_of_postings is (document_id (int), tf (int), positions (list[int])).
            Returns None if a major error occurs (e.g., file not found, struct error).
            Returns an empty list if no terms are provided or if terms are valid but have no postings (should not happen with valid offsets).
    """
    if not posting_file_path.is_file():
        logging.error(f"No such file '{posting_file_path}'!")
        return None

    result = []
    try:
        with open(posting_file_path, 'rb') as f:
            for term, (term_start_offset, term_end_offset) in zip(terms, offsets):
                postings = []

                try:
                    f.seek(term_start_offset)

                    while True:
                        current_position = f.tell()

                        if term_end_offset != -1 and current_position >= term_end_offset:
                            if current_position > term_end_offset:
                                logging.warning(f"File pointer {current_position} overshot term_end_offset {term_end_offset}.")
                            break

                        # Read the 6-byte header for posting
                        posting_header = f.read(6)

                        if not posting_header: # EOF
                            if term_end_offset != -1 and current_position < term_end_offset:
                                logging.warning(f"EOF reached unexpectedly at {current_position} before term_end_offset {term_end_offset}.")
                            break

                        if len(posting_header) < 6:
                            logging.warning(f"Incomplete posting header at {current_position}. Expected 6, got {len(posting_header)}.")
                            break

                        # Unpack document_id (4 bytes), term_frequency (2 bytes)
                        document_id, term_frequency = struct.unpack('>IH', posting_header) # Renamed positions_count to term_frequency for clarity

                        positions_data = f.read(term_frequency * 4) # tf is used as positions_count here
                        if len(positions_data) < term_frequency * 4:
                            logging.warning(f"Incomplete positions data for document_id {document_id} under term '{term}'. Expected {term_frequency * 4}, got {len(positions_data)}.")
                            break

                        positions = []
                        for i in range(term_frequency): # Iterate based on actual term_frequency
                            position_offset_in_data = i * 4
                            position = struct.unpack('>I', positions_data[position_offset_in_data : position_offset_in_data + 4])[0]
                            positions.append(position)

                        postings.append((document_id, term_frequency, positions))

                    result.append((term, postings.copy()))
                except struct.error as e:
                    logging.error(f"Error unpacking postings data for term '{term}': {e}")
                    return None # Propagate error
                except Exception as e:
                    logging.error(f"Error reading postings from file '{posting_file_path}' for term '{term}': {e}")
                    return None # Propagate error
    except Exception as e:
        logging.error(f"Error reading postings from '{posting_file_path}': {e}")
        return None

    return result

def load_document_lengths(file_path: Path) -> Union[Dict[int, int], None]:
    if not file_path.is_file():
        logging.error(f"Document lengths file not found: {file_path}")
        return None
    document_lengths: Dict[int, int] = {}
    try:
        with open(file_path, "rb") as f:
            while True:
                doc_id_bytes = f.read(4)
                if not doc_id_bytes:
                    break
                length_bytes = f.read(4)
                if not length_bytes:
                    logging.error("Malformed document lengths file: missing length for a document ID.")
                    break
                doc_id = struct.unpack('>I', doc_id_bytes)[0]
                length = struct.unpack('>I', length_bytes)[0]
                document_lengths[doc_id] = length
        logging.info(f"Loaded {len(document_lengths)} document lengths from '{file_path}'.")
        return document_lengths
    except FileNotFoundError:
        logging.error(f"Document lengths file not found at '{file_path}'.")
        return None
    except struct.error as e:
        logging.error(f"Struct unpacking error loading document lengths from '{file_path}': {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading document lengths from '{file_path}': {e}")
        return None


def main():
    args = parse_args()
    setup_logging(args)

    extract_documents(Path("/home/malik/Nextcloud/University/Semester 6/Information Retrieval/Aufgaben/collectionandqueries.tar.gz"),
                      "collection.tsv",
                      Path("/home/malik/Nextcloud/University/Semester 6/Information Retrieval/Aufgaben/output/documents"))

if __name__ == "__main__":
    main()
