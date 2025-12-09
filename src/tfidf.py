"""
TF-IDF (Term Frequency-Inverse Document Frequency) implementation.

This module provides functions to compute TF-IDF scores for terms in documents,
with support for multiple term frequency weighting schemes.
"""
# Assumptions made:
#   - Different case should not be a different word (i.e. HeLLo = hello = HELLO)
#   - Whitespace is the token delimiter (not sophisticated sentence/word boundary detection)
#   - Punctuation (except apostrophes) is removed during tokenization (see regex)
#   - Apostrophes within words are preserved (contractions), but leading/trailing apostrophes are stripped (quotes)
#   - Numbers and alphanumeric sequences are treated as valid tokens (see regex)
#   - Empty documents are valid and return 0.0 for term frequency
#   - IDF uses smoothing: log((1 + N) / (1 + n_t)) to avoid division by zero

#   - TODO: WHICH CHARACTERS SHOULD BE SUPPORTED? Emojis? what encoding?

import math
import re
from enum import StrEnum, auto


class WeightingSchemes(StrEnum):
    BINARY = auto()
    RAW_COUNT = auto()
    TERM_FREQUENCY = auto()
    LOG_NORMALISATION = auto()


# Removes punctuation but keeps apostrophes
PUNCTUATION_REGEX = r"[^\w\s']"


def tokenise_document(document: str) -> list[str]:
    """
    Tokenises a document with smart apostrophe handling.

    Preserves apostrophes within words (contractions like "don't") but removes
    surrounding quotes/apostrophes used for quotation (like 'hello').

    The tokenisation process:
    1. Removes all punctuation except apostrophes (to preserve contractions)
    2. Converts all text to lowercase for case-insensitive matching
    3. Splits on whitespace to create individual tokens
    4. Strips leading/trailing apostrophes from each token (removes quote marks)
    5. Filters out any empty tokens

    Args:
        document: The text document to tokenise.

    Returns:
        A list of lowercase tokens (words) from the document.

    Example:
        >>> tokenise_document("Hello, World! It's a test.")
        ['hello', 'world', "it's", 'a', 'test']
        >>> tokenise_document("I couldn't say 'hello' to them")
        ['i', "couldn't", 'say', 'hello', 'to', 'them']
    """
    clean_doc = re.sub(PUNCTUATION_REGEX, "", document).lower()
    tokens = clean_doc.split()
    cleaned_tokens = [token.strip("'") for token in tokens]
    return [token for token in cleaned_tokens if token]


def compute_term_frequency(term: str, tokenised_document: list[str], weighting_scheme: WeightingSchemes) -> float:
    """
    Compute term frequency using the specified weighting scheme.

    Args:
        term: The term to compute frequency for (case-insensitive).
        tokenised_document: A list of tokens from the document.
        weighting_scheme: The weighting scheme to use:
            - BINARY: Returns 1.0 if term exists, 0.0 otherwise
            - RAW_COUNT: Returns the raw count of term occurrences
            - TERM_FREQUENCY: count / doc_length
            - LOG_NORMALISATION: log(1 + count)

    Returns:
        The computed term frequency as a float.

    Example:
        >>> doc = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        >>> compute_term_frequency('the', doc, WeightingSchemes.TERM_FREQUENCY)
        0.3333333333333333
    """
    if not tokenised_document:
        return 0.0

    term = term.lower()
    raw_count = tokenised_document.count(term)

    match weighting_scheme:
        case WeightingSchemes.BINARY:
            return 1.0 if raw_count > 0 else 0.0
        case WeightingSchemes.RAW_COUNT:
            return float(raw_count)
        case WeightingSchemes.TERM_FREQUENCY:
            return raw_count / len(tokenised_document)
        case WeightingSchemes.LOG_NORMALISATION:
            return math.log(1 + raw_count)
        case _:
            raise ValueError(
                f"Unknown weighting scheme: {weighting_scheme}. Valid options: {', '.join(ws.value for ws in WeightingSchemes)}"
            )


def compute_inverse_document_frequency(term: str, tokenised_corpus: list[list[str]]) -> float:
    """
    Compute inverse document frequency for a term across a corpus.

    IDF measures how rare or common a term is across all documents. Terms that appear
    in many documents have lower IDF scores, while rare terms have higher scores.

    Formula: `log((1 + N) / (1 + n_t))`
             where N = total documents, n_t = documents containing term

    The +1 smoothing prevents division by zero

    Args:
        term: The term to compute IDF for (case-insensitive).
        tokenised_corpus: A list of tokenised documents (list of token lists).

    Returns:
        The inverse document frequency as a float. Higher values indicate rarer terms.

    Example:
        >>> corpus = [['cat', 'sat'], ['dog', 'sat'], ['cat', 'ran']]
        >>> compute_inverse_document_frequency('cat', corpus)
        0.28768207245178085  # appears in 2/3 documents
    """
    term = term.lower()
    num_docs_with_term = 0
    for tokenised_document in tokenised_corpus:
        if term in tokenised_document:
            num_docs_with_term += 1

    return math.log((1 + len(tokenised_corpus)) / (1 + num_docs_with_term))


def compute_tfidf(
    term: str,
    document: str | list[str],
    corpus: list[str] | list[list[str]],
    tf_weighting_scheme: WeightingSchemes = WeightingSchemes.TERM_FREQUENCY,
) -> float:
    """
    Compute TF-IDF score for a term in a document within a corpus.

    TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that
    reflects how important a word is to a document in a collection of documents.
    It increases proportionally with the number of times a word appears in the document
    but is offset by the frequency of the word in the corpus.

    Formula: `TF-IDF = TF(term, document) * IDF(term, corpus)`
             where TF = Term Frequency and IDF = Inverse Document Frequency

    Note:
        For repeated queries on the same corpus, pre-tokenise the corpus once
        and pass the tokenised version to avoid redundant tokenisation overhead.

    Args:
        term: The term to compute TF-IDF for (case-insensitive).
        document: Raw document string OR pre-tokenised list of terms.
        corpus: List of raw document strings OR list of pre-tokenised documents.
        tf_weighting_scheme: The term frequency weighting scheme to use.
            Defaults to TERM_FREQUENCY (normalised count).

    Returns:
        The TF-IDF score as a float. Higher scores indicate terms that are important
        to the specific document but rare across the corpus.

    Example:
        >>> corpus = [
        ...     "the cat sat on the mat",
        ...     "the dog sat on the log",
        ...     "cats and dogs are pets"
        ... ]
        >>> compute_tfidf("cat", corpus[0], corpus)
        0.060959898847262366

        >>> # Using pre-tokenised for efficiency
        >>> doc = tokenise_document("the cat sat")
        >>> tokenised_corpus = [tokenise_document(d) for d in corpus]
        >>> compute_tfidf("cat", doc, tokenised_corpus)
        0.048
    """
    # Handle both string and pre-tokenised inputs
    tokenised_document: list[str] = tokenise_document(document) if isinstance(document, str) else document
    tokenised_corpus: list[list[str]]
    tokenised_corpus = [tokenise_document(doc) for doc in corpus] if corpus and isinstance(corpus[0], str) else corpus  # type:ignore

    tf = compute_term_frequency(term, tokenised_document, tf_weighting_scheme)
    idf = compute_inverse_document_frequency(term, tokenised_corpus)

    return tf * idf
