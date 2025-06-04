from pathlib import Path
import logging
import argparse
import tarfile
import re
import struct
import bisect
from typing import Union, List, Tuple, Dict, Set
from .utils import setup_logging, load_dictionary, get_postings, load_document_lengths
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

def get_document_ids_for_phrase(
    phrase_query: str,
    dictionary_items: List[Tuple[str, int]],
    postings_file_path: Path,
    stop_words: List[str]) -> Union[Set[int], None]:
    """Helper to get document IDs for a phrase query."""
    logging.debug(f"Getting document IDs for phrase: '{phrase_query}'")

    phrase_search_results = phrase_search(phrase_query, dictionary_items, postings_file_path, stop_words)

    if phrase_search_results is None:
        return None
    if not phrase_search_results:
        return set()

    doc_ids = {match[0] for match in phrase_search_results[0][1]}
    return doc_ids

def tokenise_boolean_query(query: str) -> List[str]:
    """
    Tokenises a boolean query string into individual terms and operators.
    Operators: ! (not), ^ (and), | (or), ~ (xor), /<number> (near)
    Grouping: (, )
    Phrases: "term1 term2"
    Escape sequences: \\ (to escape special characters)

    Args:
        query (str): The boolean query string.

    Returns:
        List[str]: A list of tokens (terms and operators).
    """
    tokens = []
    current_term_characters = []
    i = 0
    n = len(query)
    char_operators_and_parentheses = ['(', ')', '!', '^', '|', '~']

    while i < n:
        char = query[i]

        if char == '"':  # Start of a phrase
            if current_term_characters:
                tokens.append(''.join(current_term_characters))
                current_term_characters = []
            
            phrase_content_characters = []
            i += 1
            phrase_closed = False
            while i < n:
                phrase_character = query[i]
                if phrase_character == '\\':  # Escape within phrase
                    if i + 1 < n:
                        # Add the escaped character (e.g., " or \) literally
                        phrase_content_characters.append(query[i + 1])
                        i += 2
                    else:  # Dangling escape at end of phrase/query
                        phrase_content_characters.append('\\') # Treat as literal backslash
                        i += 1
                elif phrase_character == '"':  # End of phrase
                    tokens.append(f'"{''.join(phrase_content_characters)}"')
                    i += 1
                    phrase_closed = True
                    break
                else:
                    phrase_content_characters.append(phrase_character)
                    i += 1
            
            if not phrase_closed:
                logging.warning("Unterminated phrase in query. Treating the content as a literal term including the initial quote.")
                tokens.append(f'"{''.join(phrase_content_characters)}')
            continue

        elif char == '\\':  # Escape character (outside of a phrase)
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
        elif char in char_operators_and_parentheses:
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
                logging.warning(f"Character '/' at position {i} not forming a valid NEAR operator. Treating as term character.")
                current_term_characters.append(char) # Treat as part of a term
                i += 1
        elif char.isspace():
            if current_term_characters:
                tokens.append(''.join(current_term_characters))
                current_term_characters = []
            i += 1
        else: # Term character
            current_term_characters.append(char)
            i += 1

    if current_term_characters:
        tokens.append(''.join(current_term_characters))

    return [token for token in tokens if token] # Filter out empty tokens

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
        is_standard_operator = token in standard_operators
        is_near_operatorerator = token.startswith('/') and len(token) > 1 and token[1:].isdigit()
        is_operator_char = is_standard_operator or is_near_operatorerator

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
            operator_key = '/' if is_near_operatorerator else token
            
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
    universal_document_ids: Set[int],
    postings_file_path: Path = DEFAULT_OUTPUT_DIR / "postings",
    stop_words: List[str] = STOPWORDS
) -> Union[Set[int], None]:
    if tokens is None:
        return None
    logging.debug(f"Evaluating RPN expression: {' '.join(tokens)}")

    operand_stack: List[Union[str, Set[int]]] = []

    # Helper to resolve an item from stack
    def _resolve_to_set(item: Union[str, Set[int]], operation_name: str) -> Union[Set[int], None]:
        if isinstance(item, set):
            return item
        if isinstance(item, str):
            if item.startswith('"') and item.endswith('"') and len(item) > 1:
                phrase_content = item[1:-1] # Extract content without quotes
                docs = get_document_ids_for_phrase(phrase_content, dictionary_items, postings_file_path, stop_words)
            else:
                docs = get_document_ids_for_term(item, dictionary_items, postings_file_path)
            
            if docs is None:
                logging.error(f"Error fetching doc IDs for operand '{item}' (for {operation_name}) in RPN eval.")
                return None
            return docs
        logging.error(f"Invalid type on RPN stack for {operation_name}: {type(item)}. Expected str or set.")
        return None

    for token in tokens:
        is_standard_operatorerator = token in ['!', '^', '|', '~']
        is_near_operatorerator = token.startswith('/') and len(token) > 1 and token[1:].isdigit()
        is_operator = is_standard_operatorerator or is_near_operatorerator

        if not is_operator:  # Operand
            operand_stack.append(token)
        
        elif is_near_operatorerator:
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
            
            right_operator = operand_stack.pop()
            left_operator = operand_stack.pop()

            if not (isinstance(left_operator, str) and not (left_operator.startswith('"') and left_operator.endswith('"')) and
                    isinstance(right_operator, str) and not (right_operator.startswith('"') and right_operator.endswith('"'))):
                logging.error(f"NEAR operator '{token}' currently only supports simple (non-phrase) term operands. Got: {type(left_operator)}, {type(right_operator)}")
                return None 
            
            termA, termB = left_operator.lower(), right_operator.lower()

            postingsA = direct_search(termA, dictionary_items, postings_file_path)
            if postingsA is None: return None
            positionsA: Dict[int, List[int]] = {}
            if postingsA:
                positionsA = {doc_id: positions_list for doc_id, _, positions_list in postingsA[0][1]}

            postingsB = direct_search(termB, dictionary_items, postings_file_path)
            if postingsB is None: return None
            positionsB: Dict[int, List[int]] = {}
            if postingsB:
                positionsB = {doc_id: positions_list for doc_id, _, positions_list in postingsB[0][1]}
            
            if not positionsA or not positionsB:
                operand_stack.append(set())
                continue

            common_document_ids = set(positionsA.keys()).intersection(positionsB.keys())
            near_documents_result = set()

            for document_id_near in common_document_ids:
                positions_listA = positionsA[document_id_near]
                positions_listB = positionsB[document_id_near]
                found_near_in_document = False
                for positionA in positions_listA:
                    for positionB in positions_listB:
                        distance = abs(positionA - positionB)
                        if 1 <= distance <= k:
                            near_documents_result.add(document_id_near)
                            found_near_in_document = True
                            break
                    if found_near_in_document:
                        break
            operand_stack.append(near_documents_result)

        elif token == '!':
            if not operand_stack:
                logging.error("Not operator '!' encountered with no operands in stack.")
                return None
            operand_item_to_negate = operand_stack.pop()
            operand_set_to_negate = _resolve_to_set(operand_item_to_negate, "NOT")
            if operand_set_to_negate is None: return None
            if universal_document_ids is None:
                logging.error("Cannot perform NOT: universal set of documents not provided (is None).")
                return None 
            result = universal_document_ids - operand_set_to_negate
            operand_stack.append(result)
            
        else: # Binary operators: ^, |, ~
            if len(operand_stack) < 2:
                logging.error(f"Operator '{token}' encountered with insufficient operands in stack.")
                return None
            right_operator = operand_stack.pop()
            left_operator = operand_stack.pop()

            left_set = _resolve_to_set(left_operator, token)
            right_set = _resolve_to_set(right_operator, token)

            if left_set is None or right_set is None: return None

            if token == '^':  # AND
                operand_stack.append(left_set.intersection(right_set))
            elif token == '|': # OR
                operand_stack.append(left_set.union(right_set))
            elif token == '~': # XOR
                operand_stack.append(left_set.symmetric_difference(right_set))
            else: # Unknown operator
                logging.error(f"Unknown operator '{token}' encountered in RPN evaluation logic.")
                return None

    if len(operand_stack) == 1:
        final_item = operand_stack[0]
        final_set = _resolve_to_set(final_item, "final result")
        if final_set is None:
             logging.error("Failed to resolve final RPN stack item to a document set.")
             return None
        return final_set
    elif not operand_stack and not tokens:
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
                    universal_document_ids: Set[int] = None,
                    stop_words: List[str] = STOPWORDS) -> Union[Set[int], None]:

    """
    Searches for a boolean query in the postings file using the dictionary.
    The query is expected to be a sequence of terms and operators in infix notation.
    Operators: ! (not), ^ (and), | (or), ~ (xor), /<number> (near)
    Grouping: (, )
    Escape sequences: \\ (to escape special characters)

    Args:
        query (str): The boolean query string.
        dictionary_items (List[Tuple[str, int]]): Dictionary items (term, offset_in_postings_file), 
                                                         sorted lexicographically by term.
        postings_file_path (Path): Path to the postings file.
        universal_document_ids (Set[int]): Optional set of document IDs to limit the search space.

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
        return []

    try:
        tokens = tokenise_boolean_query(query)
        if not tokens:
            logging.warning(f"Query '{query}' tokenisation resulted in no valid tokens.")
            return []

        original_operands = []
        for t in tokens:
            is_standard_operator = t in ['!', '^', '|', '~']
            is_near_operator = t.startswith('/') and len(t) > 1 and t[1:].isdigit()
            is_parentheses = t in ['(', ')']
            if not (is_standard_operator or is_near_operator or is_parentheses):
                original_operands.append(t)

        rpn_tokens = to_reverse_polish_notation(tokens)
        if rpn_tokens is None:
            logging.error(f"Failed to convert query '{query}' to RPN.")
            return [] 
        if not rpn_tokens:
            return []

        active_universal_set = universal_document_ids if universal_document_ids is not None else set()
        
        result_document_ids = evaluate_rpn(rpn_tokens, dictionary_items, active_universal_set, postings_file_path, stop_words)

        if result_document_ids is None:
            logging.error(f"Error evaluating RPN for query '{query}'.")
            return None 
        if not result_document_ids:
            logging.debug(f"Boolean query '{query}' yielded no matching documents.")
            return []

        results_for_ranking: List[Tuple[str, List[Tuple[int, int, List[int]]]]] = []
        
        unique_operands = sorted(list(set(original_operands)))

        for operand_token in unique_operands:
            operand_matches = []
            term_or_phrase_for_key = ""

            if operand_token.startswith('"') and operand_token.endswith('"') and len(operand_token) > 1:
                phrase_content = operand_token[1:-1]
                term_or_phrase_for_key = operand_token # Use the quoted form as the "term" key for BM25
                phrase_search_result_list = phrase_search(phrase_content, dictionary_items, postings_file_path, stop_words)
                if phrase_search_result_list:
                    all_phrase_matches = phrase_search_result_list[0][1]
                    for document_id, freq, positions_list in all_phrase_matches:
                        if document_id in result_document_ids:
                            operand_matches.append((document_id, freq, positions_list))
            else:
                term_to_search = operand_token.lower()
                term_or_phrase_for_key = term_to_search
                direct_search_result_list = direct_search(term_to_search, dictionary_items, postings_file_path)
                if direct_search_result_list:
                    all_term_postings = direct_search_result_list[0][1]
                    for document_id, tf, positions_list in all_term_postings:
                        if document_id in result_document_ids:
                            operand_matches.append((document_id, tf, positions_list))
            
            if operand_matches:
                operand_matches.sort(key=lambda x: x[0])
                results_for_ranking.append((term_or_phrase_for_key, operand_matches))
        
        if not results_for_ranking and result_document_ids:
            logging.debug(f"Boolean search for '{query}' yielded {len(result_document_ids)} documents, but no scorable terms/phrases for BM25.")
        
        return results_for_ranking

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

    query = query.strip()
    if not query:
        logging.warning("Empty query provided.")
        return []

    results: Union[List[Tuple[str, List[Tuple[int, int, List[int]]]]], None] = None

    # Check if the query is a boolean query
    if is_boolean_query(query):
        logging.debug(f"Query '{query}' identified as a boolean query.")
        # Pass `document_ids` as `universal_document_ids` and `stop_words` as `stop_words`
        results = boolean_search(query, dictionary_items, postings_file_path, 
                                 document_ids, stop_words)

    else:
        if '"' in query and query.startswith('"') and query.endswith('"'): # Quoted phrase
            phrase_content = query.strip('"')

        # Check if the query is a phrase
        if ' ' in query.strip() and not query.strip().endswith('*'):
            logging.debug(f"Query '{query}' identified as a phrase search.")
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
    #                  Path("/home/malik/Nextcloud/University/Semester 6/Information Retrieval/Aufgaben/output/documents",
    #                  replace=True))

    calculate_and_save_document_lengths(
        source_path=Path("/home/malik/Nextcloud/University/Semester 6/Information Retrieval/Aufgaben/collectionandqueries.tar.gz"),
        content_file_name="collection.tsv")
    
    return 0

if __name__ == "__main__":
    main()
