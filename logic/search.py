from pathlib import Path
import logging
import argparse
import tarfile
import heapq
import re
import shutil
import struct
import io
import bisect
from typing import Union, List, Tuple, Dict, Set
from .utils import setup_logging
import math

WDIR = Path(__file__).resolve().parent.parent / "static"
DEFAULT_OUTPUT_DIR = WDIR / "index"

STOPWORDS = ["the", "is", "a", "an", "and", "or", "of"]

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
                                      output_dir: Path = DEFAULT_OUTPUT_DIR,
                                      postings_file_name: str = "postings", replace: bool = False) -> int:
    """
    Builds the postings file and dictionary using a two-pass approach (SPIMI-like).
    First pass: Creates temporary blocks of inverted indexes.
    Second pass: Merges blocks into a final postings file and creates the dictionary.

    Args:
        source_path (Path): Path to the source archive.
        content_file_name (str): Name of the content file inside the archive.
        stop_words (list[str]): List of stop words to exclude.
        output_dir (Path): Directory to save output files.
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

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / postings_file_name
    dictionary_file_path = output_dir / f"{postings_file_name}_dictionary"
    temp_dir = output_dir / "temp_postings"

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

def calculate_and_save_document_lengths(source_path: Path, content_file_name: str,
                                        output_dir: Path = DEFAULT_OUTPUT_DIR,
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
        output_dir (Path): Directory to save the output binary file.
        output_file_name (str): Name for the output binary file.
        replace (bool): Whether to overwrite the output file if it exists.

    Returns:
        int: 0 on success, 1 on failure.
    """
    logging.debug(f"Calculating document lengths from '{source_path}/{content_file_name}' for binary output.")

    if not source_path.is_file():
        logging.error(f"No such file '{source_path}'!")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / output_file_name

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

def direct_search(term: str,
                  dictionary_items: List[Tuple[str, int]],
                  postings_file_path: Path = DEFAULT_OUTPUT_DIR / "postings") -> Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None]:
    """
    Retrieves all postings for a given search term using the dictionary.

    Args:
        term (str): The search term.
        dictionary_items (List[Tuple[str, int]]): The pre-loaded dictionary items 
                                                         (term, offset_in_postings_file), sorted by term.
        postings_file_path (Path): Path to the postings file.

    Returns:
        Union[List[Tuple[int, int, List[int]]], None]:
            A list of postings for the term. Each posting is (document_id, term_frequency, positions).
            Returns an empty list ([]) if the term is not found.
            Returns None if an error occurs during postings retrieval.
    """
    logging.debug(f"Getting document IDs for term '{term}'")

    if not postings_file_path.is_file():
        logging.error(f"Postings file '{postings_file_path}' does not exist!")
        return None

    term = term.lower()

    idx = bisect.bisect_left(dictionary_items, term, key=lambda item: item[0])

    term_start_offset = -1
    term_found = False

    if idx < len(dictionary_items) and dictionary_items[idx][0] == term:
        term_start_offset = dictionary_items[idx][1]
        term_found = True
    
    if not term_found:
        logging.debug(f"Term '{term}' not found in dictionary.")
        return []

    term_end_offset = -1
    if idx + 1 < len(dictionary_items):
        # The end_offset for the current term is the start_offset of the next term
        term_end_offset = dictionary_items[idx+1][1]
    # If it's the last term in the dictionary, term_end_offset remains -1 (read to EOF)

    postings_result = get_postings(postings_file_path, [(term_start_offset, term_end_offset)], [term])

    if postings_result is None:
        return None
    if not postings_result: # Should not happen if term was in dictionary
        logging.warning(f"direct_search: get_postings returned empty for found term '{term}'.")
        return []

    return postings_result

def load_document_lengths_from_file(file_path: Path) -> Union[Dict[int, int], None]:
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

def get_unique_document_ids(postings_file_path: Path) -> Set[int]:
    """
    Retrieves all unique document IDs from the postings file.

    Args:
        postings_file_path (Path): Path to the postings file.

    Returns:
        Set[int]: A set of unique document IDs across all terms in the postings file.
                  Returns an empty set if no documents are found.
                  Returns None if an error occurs during postings retrieval.
    """
    logging.debug("Getting unique document IDs from postings file.")

    if not postings_file_path.is_file():
        logging.error(f"Postings file '{postings_file_path}' does not exist!")
        return None

    unique_document_ids = set()

    try:
        with open(postings_file_path, 'rb') as f:
            while True:
                # Read the 6-byte header for posting
                posting_header = f.read(6)
                if not posting_header:  # EOF
                    break
                if len(posting_header) < 6:
                    logging.warning("Incomplete posting header found. Stopping reading postings.")
                    break

                # Unpack document_id (4 bytes), term_frequency (2 bytes)
                document_id, term_frequency = struct.unpack('>IH', posting_header)

                unique_document_ids.add(document_id)

                # Skip to next document
                f.seek(term_frequency * 4, 1)
    except IOError as e:
        logging.error(f"IOError reading postings file '{postings_file_path}': {e}")
        return None
    except struct.error as e:
        logging.error(f"Struct unpacking error reading postings file '{postings_file_path}': {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error reading postings file '{postings_file_path}': {e}")
        return None

    return unique_document_ids

def get_document_ids_for_term(term: str, dictionary_items: List[Tuple[str, int]], postings_file_path: Path) -> Set[int]:
    """
    Retrieves document IDs for a given term using the dictionary and postings file.

    Args:
        term (str): The search term.
        dictionary_items (List[Tuple[str, int]]): The pre-loaded dictionary items 
                                                         (term, offset_in_postings_file), sorted by term.
        postings_file_path (Path): Path to the postings file.

    Returns:
        Set[int]: A set of document IDs where the term appears.
                  Returns an empty set if the term is not found.
                  Returns None if an error occurs during postings retrieval.
    """
    term_postings = direct_search(term, dictionary_items, postings_file_path)
    if not term_postings or not term_postings[0][1]:
        logging.debug(f"No postings found for term '{term}'.")
        return set()
    postings_list = term_postings[0][1] # [(document_id, term_frequency, [positions])]
    return {document_id for document_id, _, _ in postings_list}

def tokenise_boolean_query(query: str) -> List[str]:
    """
    Tokenises a boolean query string into individual terms and operators.
    Operators: ! (not), ^ (and), | (or), ~ (xor), /<number> (near)
    Grouping: (, )
    Escape sequences: \ (to escape special characters)

    Args:
        query (str): The boolean query string.

    Returns:
        List[str]: A list of tokens (terms and operators).
    """
    tokens = []
    current_term_characters = []
    i = 0
    n = len(query)

    while i < n:
        char = query[i]

        if char == '\\':  # Escape character
            if current_term_characters:
                tokens.append(''.join(current_term_characters))
                current_term_characters = []
            if i + 1 < n:
                current_term_characters.append(query[i + 1])
                i += 2
            else:
                logging.warning("Trailing escape character at the end of query. Treating as literal '\\'.")
                current_term_characters.append('\\')
                i += 1
        elif char in ('(', ')', '!', '^', '|', '~'):
            if current_term_characters:
                tokens.append(''.join(current_term_characters))
                current_term_characters = []
            tokens.append(char)
            i += 1
        elif char == '/': # Potential NEAR operator /<number>
            if current_term_characters:
                tokens.append(''.join(current_term_characters))
                current_term_characters = []
            if i + 1 < n and query[i + 1].isdigit():
                j = i + 1
                while j < n and query[j].isdigit():
                    j += 1
                tokens.append(query[i:j])
                i = j
            else:
                # If '/' is not part of /<number>, it's treated as a term character by the else block below
                # or you can explicitly log a warning and ignore it as per your original code:
                logging.warning(f"Character '/' at position {i} not forming a valid NEAR operator. Treating as term character.")
                current_term_characters.append(char) # Treat as part of a term
                i += 1
                #
                #
                #
        elif char.isspace(): # Space delimiter
            if current_term_characters:
                tokens.append(''.join(current_term_characters))
                current_term_characters = []
            i += 1
        else: # Term character
            current_term_characters.append(char)
            i += 1

    if current_term_characters:
        tokens.append(''.join(current_term_characters))

    # Filter out empty string tokens
    return [token for token in tokens if token]

def to_reverse_polish_notation(tokens: List[str]) -> List[str]:
    """
    Converts a list of infix tokens to Reverse Polish Notation (RPN) using the Shunting Yard algorithm.

    Args:
        tokens (List[str]): A list of tokens from a boolean query.

    Returns:
        List[str]: A list of tokens in Reverse Polish Notation (RPN).
    """
    precedence = {'!': 4, '/': 3, '^': 2, '|': 1, '~': 1}
    associativity = {'!': 'R', '/': 'L', '^': 'L', '|': 'L', '~': 'L'}

    output_queue: List[str] = []
    operator_stack: List[str] = []

    standard_operators = ['!', '^', '|', '~']

    for token in tokens:
        is_std_op = token in standard_operators
        is_near_op = token.startswith('/') and len(token) > 1 and token[1:].isdigit()
        is_operator_char = is_std_op or is_near_op

        if not is_operator_char and token not in ['(', ')']:  # Operand
            output_queue.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())

            if not operator_stack or operator_stack[-1] != '(':
                logging.error("Mismatched parentheses in query (extra ')' or missing '('). Aborting conversion to RPN.")
                return None
            operator_stack.pop()  # Pop '('
        else:
            operator_key = '/' if is_near_op else token
            
            while (operator_stack and operator_stack[-1] != '(' and
                   (precedence.get(operator_stack[-1], 0) > precedence.get(operator_key, 0) or
                    (precedence.get(operator_stack[-1], 0) == precedence.get(operator_key, 0) and
                     associativity.get(operator_key, 'L') == 'L'))):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)

    while operator_stack:
        if operator_stack[-1] == '(':
            logging.error("Mismatched parentheses in query. Aborting conversion to RPN.")
            return None
        output_queue.append(operator_stack.pop())
    
    logging.debug(f"RPN: {' '.join(output_queue if output_queue is not None else [])}")
    return output_queue

def evaluate_rpn(
    tokens: List[str],
    dictionary_items: List[Tuple[str, int]],
    document_ids: Set[int],
    postings_file_path: Path = DEFAULT_OUTPUT_DIR / "postings") -> Union[Set[int], None]:
    """
    Evaluates a Reverse Polish Notation (RPN) expression against the postings file.

    Args:
        tokens (List[str]): The RPN expression tokens.
        dictionary_items (List[Tuple[str, int]]): The pre-loaded dictionary items 
                                                         (term, offset_in_postings_file), sorted by term.
        document_ids (Set[int]): The set of document IDs to consider for the evaluation.
        postings_file_path (Path): Path to the postings file.

    Returns:
        Union[Set[int], None]: The set of document IDs that match the RPN expression.
                                Returns None if an error occurs during evaluation.
    """
    logging.debug(f"Evaluating RPN expression: {' '.join(tokens)}")

    if not tokens:
        logging.error("Empty or None RPN tokens provided for evaluation.")
        return None

    operand_stack: List[Union[str, Set[int]]] = []

    # Helper to resolve an item from stack (str or Set[int]) to Set[int]
    def _resolve_to_set(item: Union[str, Set[int]], operation_name: str) -> Union[Set[int], None]:
        if isinstance(item, set):
            return item
        if isinstance(item, str):
            documents = get_document_ids_for_term(item, dictionary_items, postings_file_path)
            if documents is None:
                logging.error(f"Error fetching doc IDs for term '{item}' (operand for {operation_name}) in RPN eval.")
                return None
            return documents
        logging.error(f"Invalid type on RPN stack for {operation_name}: {type(item)}. Expected str or set.")
        return None

    for token in tokens:
        is_standard_operator = token in ['!', '^', '|', '~']
        is_near_operator = token.startswith('/') and len(token) > 1 and token[1:].isdigit()
        is_operator = is_standard_operator or is_near_operator

        if not is_operator:  # Operand
            operand_stack.append(token.lower())
        
        elif is_near_operator:
            try:
                k = int(token[1:])
                if k < 1: 
                    logging.error(f"Invalid k value for NEAR operator: {token}. k must be >= 1.")
                    return None
            except ValueError:
                logging.error(f"Malformed NEAR operator: {token}")
                return None

            if len(operand_stack) < 2:
                logging.error(f"NEAR operator '{token}' needs two term operands on stack.")
                return None
            
            operand_right = operand_stack.pop()
            operand_left = operand_stack.pop()

            if not (isinstance(operand_left, str) and isinstance(operand_right, str)):
                logging.error(f"NEAR operator '{token}' received non-term operands. Operands must be terms. Got: {type(operand_left)}, {type(operand_right)}")
                return None 
            
            termA_str, termB_str = operand_left, operand_right

            postingsA_data = direct_search(termA_str, dictionary_items, postings_file_path)
            if postingsA_data is None: return None
            termA_positions_map: Dict[int, List[int]] = {} # document_id -> [positions]
            if postingsA_data:
                termA_positions_map = {document_id: pos_list for document_id, _, pos_list in postingsA_data[0][1]}

            postingsB_data = direct_search(termB_str, dictionary_items, postings_file_path)
            if postingsB_data is None: return None
            termB_positions_map: Dict[int, List[int]] = {} # document_id -> [positions]
            if postingsB_data:
                termB_positions_map = {document_id: pos_list for document_id, _, pos_list in postingsB_data[0][1]}
            
            if not termA_positions_map or not termB_positions_map: # One or both terms not found or have no postings
                operand_stack.append(set()) # NEAR results in empty set if a term is missing
                continue

            common_document_ids = set(termA_positions_map.keys()).intersection(termB_positions_map.keys())
            near_documents_result = set()

            for doc_id in common_document_ids:
                pos_listA = termA_positions_map[doc_id]
                pos_listB = termB_positions_map[doc_id]
                found_near_in_doc = False
                for pA in pos_listA:
                    for pB in pos_listB:
                        distance = abs(pA - pB)
                        if 1 <= distance <= k:
                            near_documents_result.add(doc_id)
                            found_near_in_doc = True
                            break
                    if found_near_in_doc:
                        break
            operand_stack.append(near_documents_result)

        elif token == '!': # Unary NOT
            if not operand_stack:
                logging.error("Not operator '!' encountered with no operands in stack.")
                return None
            
            operand_item_to_negate = operand_stack.pop()
            operand_set_to_negate = _resolve_to_set(operand_item_to_negate, "NOT")
            if operand_set_to_negate is None: return None

            if document_ids is None: # Should be checked by caller
                logging.error("Cannot perform NOT: universal set of documents not provided (is None).")
                return None 
            
            result = document_ids - operand_set_to_negate
            operand_stack.append(result)
            
        else: # Binary operators: ^, |, ~
            if len(operand_stack) < 2:
                logging.error(f"Operator '{token}' encountered with insufficient operands in stack.")
                return None
            
            operand_right = operand_stack.pop()
            operand_left = operand_stack.pop()

            # Resolve operands to Set[int] if they are strings
            op1_set = _resolve_to_set(operand_left, token)
            op2_set = _resolve_to_set(operand_right, token)

            if op1_set is None or op2_set is None: return None

            if token == '^':  # AND
                operand_stack.append(op1_set.intersection(op2_set))
            elif token == '|':  # OR
                operand_stack.append(op1_set.union(op2_set))
            elif token == '~':  # XOR
                operand_stack.append(op1_set.symmetric_difference(op2_set))
            else:
                logging.error(f"Unknown operator '{token}' encountered in RPN evaluation logic.")
                return None

    # Final result processing
    if len(operand_stack) == 1:
        final_item = operand_stack[0]
        final_set = _resolve_to_set(final_item, "final result")
        if final_set is None:
             logging.error("Failed to resolve final RPN stack item to a document set.")
             return None
        return final_set
    elif not operand_stack and not tokens: # Empty query
        return set()
    else:
        logging.error(f"RPN evaluation ended with invalid stack size: {len(operand_stack)}. Stack: {operand_stack}")
        return None

def is_boolean_query(query: str) -> bool:
    """
    Checks if the given query is a boolean query.

    Args:
        query (str): The query string to check.

    Returns:
        bool: True if the query is a boolean query, False otherwise.
    """
    pattern = r"(?<!\\)([!^|~()]|/(\d+))"
    if re.search(pattern, query):
            if re.search(r"(?<!\\)([!^|~]|/(\d+))", query):
                return True
    return False

def boolean_search(query: str,
                    dictionary_items: List[Tuple[str, int]],
                    postings_file_path: Path = DEFAULT_OUTPUT_DIR / "postings",
                    document_ids: Set[int] = None) -> Union[Set[int], None]:

    """
    Searches for a boolean query in the postings file using the dictionary.
    The query is expected to be a sequence of terms and operators in infix notation.
    Operators: ! (not), ^ (and), | (or), ~ (xor), /<number> (near)
    Grouping: (, )
    Escape sequences: \ (to escape special characters)

    Args:
        query (str): The boolean query string.
        dictionary_items (List[Tuple[str, int]]): Dictionary items (term, offset_in_postings_file), 
                                                         sorted lexicographically by term.
        postings_file_path (Path): Path to the postings file.
        document_ids (Set[int]): Optional set of document IDs to limit the search space.

    Returns:
        Union[Set[int], None]: 
            A set of document IDs that match the boolean query.
            Returns an empty set if no documents match the query.
            Returns None if a major error occurs (e.g., file not found, struct error).
    """
    logging.debug(f"Searching for boolean query '{query}'")
    query = query.strip()

    if not query:
        logging.warning("Empty query provided.")
        return set()

    try:
        tokens = tokenise_boolean_query(query)
        if not tokens:
            logging.warning(f"Query '{query}' tokenisation resulted in no valid tokens.")
            return set()
        
        original_terms = set([token.lower() for token in tokens if token not in ['!', '^', '|', '~', '(', ')'] and not token.startswith('/')])

        rpn_tokens = to_reverse_polish_notation(tokens)
        if not rpn_tokens and original_terms:
            logging.warning(f"Query '{query}' resulted in empty RPN despite terms. Check for balanced operators.")
        if not rpn_tokens and not original_terms:
            return set()  # Empty query
        
        if not document_ids:
            logging.warning("No document IDs provided for boolean search. Unable to perform NOT operations.")

        result_set = evaluate_rpn(rpn_tokens, dictionary_items, document_ids or set(), postings_file_path)

        if result_set is None:
            logging.error(f"Error evaluating RPN for query '{query}'.")
            return None

        results_for_ranking: List[Tuple[str, List[Tuple[int, int, List[int]]]]] = []

        for term in original_terms:
            term_postings = direct_search(term, dictionary_items, postings_file_path)

            if term_postings and term_postings[0][1]:
                term_name, postings_list = term_postings[0]

                filtered_postings = [(doc_id, tf, positions) for doc_id, tf, positions in postings_list if doc_id in result_set]

                if filtered_postings:
                    results_for_ranking.append((term_name, filtered_postings))

        if not results_for_ranking and result_set:
            logging.debug(f"Boolean search yielded documents, but no terms for BM25 scoring matched those documents.")
        
        return results_for_ranking if results_for_ranking else set()
    except ValueError as e:
        logging.error(f"Boolean query processing error for '{query}': {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error processing boolean query '{query}': {e}")
        return None

def prefix_search(term_prefix: str,
                  dictionary_items: List[Tuple[str, int]],
                  postings_file_path: Path = DEFAULT_OUTPUT_DIR / "postings") -> Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None]:
    """
    Retrieves postings for all terms starting with the given prefix using a sorted dictionary.
    This function already primarily uses dictionary_items.

    Args:
        term_prefix (str): The prefix to search for.
        dictionary_items (List[Tuple[str, int]]): Dictionary items (term, offset_in_postings_file), 
                                                         sorted lexicographically by term.
        postings_file_path (Path): Path to the postings file.

    Returns:
        Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None]:
            A list of (matching_term, list_of_postings) tuples.
            Returns an empty list ([]) if no terms match the prefix.
            Returns None if a major error occurs.
    """
    logging.debug(f"Getting document IDs for prefix '{term_prefix}*'")
    term_prefix = term_prefix.lower()

    if not dictionary_items:
        logging.warning("Dictionary items list is empty.")
        return []

    candidate_terms_names: List[str] = []
    candidate_offsets: List[Tuple[int, int]] = []

    start_idx = bisect.bisect_left(dictionary_items, term_prefix, key=lambda item: item[0])

    for i in range(start_idx, len(dictionary_items)):
        current_term_str, current_start_offset = dictionary_items[i]

        if current_term_str.startswith(term_prefix):
            end_offset = -1 # Default to EOF for the last term matching
            if i + 1 < len(dictionary_items):
                # The end_offset is the start_offset of the next term
                _, next_term_offset = dictionary_items[i+1]
                end_offset = next_term_offset

            candidate_terms_names.append(current_term_str)
            candidate_offsets.append((current_start_offset, end_offset))
        elif candidate_terms_names:
            break
        elif not candidate_terms_names and current_term_str > term_prefix:
            # If no matches found yet and we've passed where the prefix would be alphabetically, stop.
            break


    if not candidate_terms_names:
        return []

    return get_postings(postings_file_path, candidate_offsets, candidate_terms_names)

def phrase_search(phrase: str, 
                  dictionary_items: List[Tuple[str, int]],
                  postings_file_path: Path = DEFAULT_OUTPUT_DIR / "postings",
                  stop_words: List[str] = STOPWORDS) -> Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None]: # TODO: Use a more comprehensive stop word list
    """
    Searches for a phrase in the postings file using the dictionary.
    The phrase is expected to be a sequence of words separated by spaces.

    Args:
        phrase (str): The phrase to search for.
        dictionary_items (List[Tuple[str, int]]): Dictionary items (term, offset_in_postings_file), 
                                                         sorted lexicographically by term.
        postings_file_path (Path): Path to the postings file.

    Returns:
        Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None]:
            A list containing a single tuple: (phrase_string, list_of_matches).
            Each match in list_of_matches is (document_id, phrase_frequency_in_doc, [start_positions_of_phrase]).
            Returns an empty list ([]) if the phrase is not found.
            Returns None if a major error occurs.
    """
    logging.debug(f"Searching for phrase '{phrase}'")
    
    terms = [word.lower() for word in phrase.strip().split() if word.lower() not in stop_words]
    if not terms:
        logging.warning("Empty phrase provided.")
        return []
    if len(terms) == 1:
        return direct_search(terms[0], dictionary_items, postings_file_path)

    # Fetch postings for all terms and collect their info for sorting
    term_data_for_sorting: List[Dict[str, any]] = [] # {"term": str, "document_map": Dict[int, List[int]], "document_count": int}
    terms_postings_map: Dict[str, Dict[int, List[int]]] = {} # {term_str: {document_id: [positions]}}

    for term in terms:
        term_postings_result = direct_search(term, dictionary_items, postings_file_path)

        if term_postings_result is None:
            logging.error(f"Error retrieving postings for term '{term}' in phrase '{phrase}'. Aborting phrase search.")
            return None
        if not term_postings_result:
            logging.debug(f"Term '{term}' in phrase '{phrase}' not found in index. Phrase cannot exist.")
            return []

        term_postings_list: List[Tuple[int, int, List[int]]] = term_postings_result[0][1]  # [(term_name, [(document_id, tf, [positions])])]
        
        term_document_positions: Dict[int, List[int]] = {}
        for document_id, _, positions in term_postings_list:
            term_document_positions[document_id] = positions
        
        terms_postings_map[term] = term_document_positions
        term_data_for_sorting.append({
            "term": term,
            "document_map": term_document_positions,
            "document_count": len(term_document_positions)
        })

    # Sort terms by rarity
    term_data_for_sorting.sort(key=lambda x: x["document_count"])

    # Intersect document IDs, starting with the rarest term
    if not term_data_for_sorting:
        return []
    
    candidate_document_ids = set(term_data_for_sorting[0]["document_map"].keys())

    for i in range(1, len(term_data_for_sorting)):
        candidate_document_ids.intersection_update(term_data_for_sorting[i]["document_map"].keys())
        if not candidate_document_ids:
            logging.debug(f"No common documents found for all terms in phrase '{phrase}' after intersection.")
            return bag_of_words_search(phrase, dictionary_items, postings_file_path, stop_words)
    
    # Perform positional checks on candidate_document_ids
    final_phrase_matches: List[Tuple[int, int, List[int]]] = [] # (document_id, phrase_freq, [start_positions])

    for document_id in sorted(list(candidate_document_ids)):
        document_term_positions_ordered: List[List[int]] = []
        valid_doc_for_positional_check = True
        for term in terms:
            positions = terms_postings_map[term].get(document_id)
            if positions is None: 
                # This should not happen
                logging.error(f"Internal error: Term '{term}' positions not found for document_id {document_id} which was in candidate_document_ids.")
                valid_doc_for_positional_check = False
                break
            document_term_positions_ordered.append(positions)
        
        if not valid_doc_for_positional_check:
            continue

        # Positional adjacency check
        phrase_starts_document: List[int] = []
        if not document_term_positions_ordered or not document_term_positions_ordered[0]:
            continue

        for start_position in document_term_positions_ordered[0]:
            is_match = True
            current_position_phrase = start_position
            for k in range(1, len(terms)):
                expected_next_position = current_position_phrase + 1
                
                kth_term_positions = document_term_positions_ordered[k]
                idx = bisect.bisect_left(kth_term_positions, expected_next_position)
                
                if idx == len(kth_term_positions) or kth_term_positions[idx] != expected_next_position:
                    is_match = False
                    break 
                current_position_phrase = expected_next_position
            
            if is_match:
                phrase_starts_document.append(start_position)
        
        if phrase_starts_document:
            final_phrase_matches.append((document_id, len(phrase_starts_document), sorted(phrase_starts_document)))

    if not final_phrase_matches:
        return bag_of_words_search(phrase, dictionary_items, postings_file_path, stop_words)
    
    return [(phrase, final_phrase_matches)]

def bag_of_words_search(phrase: str,
                  dictionary_items: List[Tuple[str, int]],
                  postings_file_path: Path = DEFAULT_OUTPUT_DIR / "postings",
                  stop_words: List[str] = STOPWORDS) -> Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None]:
    """
    Searches for documents containing all terms in the phrase (bag-of-words model),
    regardless of their order. Returns results formatted for BM25 ranking.

    Args:
        phrase (str): The phrase to search for.
        dictionary_items (List[Tuple[str, int]]): Dictionary items (term, offset_in_postings_file),
                                                         sorted lexicographically by term.
        postings_file_path (Path): Path to the postings file.
        stop_words (List[str]): List of stop words to exclude.

    Returns:
        Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None]:
            A list of tuples, where each tuple is (term_from_query, list_of_postings).
            Each posting is (document_id, term_frequency, [positions]), but only for documents
            that contain ALL terms from the query.
            Returns an empty list ([]) if no documents contain all terms or if the query is empty.
            Returns None if a major error occurs during postings retrieval.
    """
    logging.debug(f"Performing bag-of-words search for: '{phrase}'")

    terms = [word.lower() for word in phrase.strip().split() if word.lower() not in stop_words]
    if not terms:
        logging.warning("Empty phrase provided for bag_of_words_search after stop word removal.")
        return []
    if len(terms) == 1:
        return direct_search(terms[0], dictionary_items, postings_file_path)

    term_data_for_sorting: List[Dict[str, any]] = []
    
    all_terms_full_postings_map: Dict[str, Dict[int, Tuple[int, List[int]]]] = {}

    for term in terms:
        term_postings_result = direct_search(term, dictionary_items, postings_file_path)

        if term_postings_result is None:
            logging.error(f"Error retrieving postings for term '{term}' in bag_of_words_search for phrase '{phrase}'.")
            return None
        if not term_postings_result:
            logging.debug(f"Term '{term}' in phrase '{phrase}' not found in index. Bag of words cannot exist with all terms.")
            return []

        actual_term_postings_list = term_postings_result[0][1]
        
        current_term_doc_data: Dict[int, Tuple[int, List[int]]] = {}
        for doc_id, tf, positions_list in actual_term_postings_list:
            current_term_doc_data[doc_id] = (tf, positions_list)
        
        all_terms_full_postings_map[term] = current_term_doc_data
        term_data_for_sorting.append({
            "term": term,
            "document_map_tf_positions": current_term_doc_data,
            "doc_count": len(current_term_doc_data)
        })

    term_data_for_sorting.sort(key=lambda x: x["doc_count"])

    # Intersect document IDs, starting with the rarest term
    if not term_data_for_sorting:
        return []
    
    # Initialize intersected_document_ids with document_ids from the rarest term's postings
    intersected_document_ids = set(term_data_for_sorting[0]["document_map_tf_positions"].keys())

    for i in range(1, len(term_data_for_sorting)):
        intersected_document_ids.intersection_update(term_data_for_sorting[i]["document_map_tf_positions"].keys())
        if not intersected_document_ids:
            logging.debug(f"No common documents found for all terms in phrase '{phrase}' after intersection.")
            return []
    
    # Construct results for BM25 ranking
    results_for_ranking: List[Tuple[str, List[Tuple[int, int, List[int]]]]] = []
    
    for term_from_query in terms:
        term_specific_postings_for_ranking = []
        if term_from_query in all_terms_full_postings_map:
            term_postings_data = all_terms_full_postings_map[term_from_query]
            for doc_id in intersected_document_ids:
                if doc_id in term_postings_data:
                    tf, positions = term_postings_data[doc_id]
                    term_specific_postings_for_ranking.append((doc_id, tf, positions))
        
        if term_specific_postings_for_ranking:
            results_for_ranking.append((term_from_query, term_specific_postings_for_ranking))
            
    if not results_for_ranking:
        logging.debug(f"Bag-of-words search for '{phrase}' yielded no results for ranking, though intersection was non-empty.")
        return []
            
    return results_for_ranking

def bm25_plus_rankings(
    query_results: List[Tuple[str, List[Tuple[int, int, List[int]]]]],
    document_lengths: Dict[int, int],
    average_document_length: float,
    documents_count: int
) -> List[Tuple[int, float]]:
    """
    Ranks documents based on BM25+ scores for a given query result.

    Args:
        query_results: The query results where each tuple is (term, list_of_postings).
                       Each posting is (document_id, term_frequency, [positions]).
        document_lengths: A dictionary mapping document_id to its total length.
        average_document_length: Average document length of the entire collection.
        documents_count: Total number of documents (N) in the collection.

    Returns:
        A list of tuples where each tuple is (document_id, bm25_score), sorted by score.
    """
    if documents_count == 0 or average_document_length == 0:
        logging.warning("BM25 ranking called with zero total documents or zero average document length. Returning empty list.")
        return []
    if not query_results:
        return []

    k_1 = 1.5
    b = 0.75
    delta = 1.0

    bm25_scores: Dict[int, float] = {}

    for term, postings_list in query_results:
        if not postings_list:
            continue
        
        df_t = len(postings_list) # Documents frequency of current term
        
        idf_numerator = documents_count - df_t + 0.5
        idf_denominator = df_t + 0.5
        if idf_numerator <= 0 or idf_denominator <= 0: # Avoid math errors with extreme df_t
            idf = 0.00000001
        else:
            idf = math.log((idf_numerator / idf_denominator) + 1.0)


        for document_id, term_document_frequency, _ in postings_list:
            document_length = document_lengths.get(document_id)
            if document_length is None:
                logging.warning(f"Document ID {document_id} not found in document_lengths. Skipping for BM25.")
                continue
            if document_length == 0: # Avoid division by zero
                document_length = average_document_length

            # BM25+ term score component
            numerator = term_document_frequency * (k_1 + 1)
            denominator = term_document_frequency + k_1 * (1 - b + b * (document_length / average_document_length))
            
            term_score = idf * ((numerator / denominator if denominator != 0 else 0) + delta)
            
            bm25_scores[document_id] = bm25_scores.get(document_id, 0.0) + term_score

    return sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)

def search(
    query: str,
    dictionary_items: List[Tuple[str, int]],
    document_lengths_map: Dict[int, int],
    average_document_length: float,
    total_documents_count: int,
    document_ids: Set[int] = None,
    postings_file_path: Path = DEFAULT_OUTPUT_DIR / "postings",
    stop_words: List[str] = STOPWORDS
) -> Union[List[Tuple[int, float]], None]:
    """
    Determines the type of search, executes it, and ranks results using BM25+.

    Args:
        query (str): The search query.
        dictionary_items: Sorted dictionary items (term, offset).
        document_lengths_map: Map of document IDs to their lengths.
        average_document_length: Average document length in the collection.
        total_documents_count: Total number of documents in the collection.
        postings_file_path: Path to the postings file.
        stop_words: List of stop words.

    Returns:
        A list of (document_id, bm25_score) tuples, sorted by score, or None on error.
    """
    logging.debug(f"Searching for query '{query}'")

    if not query.strip():
        logging.warning("Empty query provided.")
        return []

    results: Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None] = None

    # Check if the query is a boolean query
    if is_boolean_query(query):
        logging.debug(f"Query '{query}' identified as a boolean query.")
        results = boolean_search(query, dictionary_items, postings_file_path, document_ids)
    # Check if the query is a phrase
    elif '"' in query and query.startswith('"') and query.endswith('"'): # Quoted phrase
        phrase_content = query.strip('"')
        results = phrase_search(phrase_content, dictionary_items, postings_file_path, stop_words)
    elif ' ' in query.strip():
        logging.debug(f"Query '{query}' identified as a phrase search or term with wildcard.")
        if not query.strip().endswith('*'):
            results = phrase_search(query, dictionary_items, postings_file_path, stop_words)
        else: # Handle "term *"
            logging.debug(f"Query '{query}' identified as a prefix search.")
            if query.endswith('*'):
                 term_prefix = query[:-1].strip()
                 results = prefix_search(term_prefix, dictionary_items, postings_file_path)
            else:
                results = phrase_search(query, dictionary_items, postings_file_path, stop_words)

    # Check if the query is a prefix search
    elif query.endswith('*'):
        logging.debug(f"Query '{query}' identified as a prefix search.")
        term_prefix = query[:-1]
        results = prefix_search(term_prefix, dictionary_items, postings_file_path)
    
    # Otherwise, perform a direct search for a single term
    else:
        logging.debug(f"Query '{query}' identified as a direct search for a single term.")
        results = direct_search(query, dictionary_items, postings_file_path)

    if results is None:
        logging.error(f"Error during search for query '{query}'.")
        return None
    
    if not results: # No initial results found
        logging.debug(f"No initial documents found for query '{query}'.")
        return []

    # Perform BM25+ ranking on the results
    ranked_results = bm25_plus_rankings(
        results,
        document_lengths_map,
        average_document_length,
        total_documents_count
    )

    # ranked_results is already List[Tuple[int, float]]
    return ranked_results


def main() -> int:
    args = parse_args()
    setup_logging(args)

    #extract_documents(Path("/home/malik/Nextcloud/University/Semester 6/Information Retrieval/Aufgaben/collectionandqueries.tar.gz"),
    #                  "collection.tsv",
    #                  Path("/home/malik/Nextcloud/University/Semester 6/Information Retrieval/Aufgaben/output/documents"))

    #calculate_and_save_document_lengths(
    #    source_path=Path("/home/malik/Nextcloud/University/Semester 6/Information Retrieval/Aufgaben/collectionandqueries.tar.gz"),
    #    content_file_name="collection.tsv")
    
    return 0

if __name__ == "__main__":
    main()