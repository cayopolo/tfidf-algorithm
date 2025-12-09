"""
TF-IDF (Term Frequency-Inverse Document Frequency) implementation.

This module provides functions to compute TF-IDF scores for terms in documents,
with support for multiple term frequency weighting schemes.
"""
# Assumptions made:
#   - Different case should not be a different word (i.e. HeLLo = hello = HELLO)
#   - TODO: WHICH CHARACTERS SHOULD BE SUPPORTED? Emojis? what encoding?
#   - TODO: Handling of aprostrophes as part of a word vs speech marks (i.e. couldn't vs 'could not')

import math
import re
from enum import StrEnum, auto


class WeightingSchemes(StrEnum):
    BINARY = auto()
    RAW_COUNT = auto()
    TERM_FREQUENCY = auto()
    LOG_NORMALISATION = auto()
    DOUBLE_NORMALISATION_K = auto()


# Removes punctuation but keeps apostrophes
PUNCTUATION_REGEX = r"[^\w\s']"


def tokenise_document(document: str) -> list[str]:
    """
    Tokenises a document by removing punctuation and splitting by whitespace (preserving apostrophes).

    The tokenisation process:
    1. Removes all punctuation except apostrophes (to preserve contractions like "don't")
    2. Converts all text to lowercase for case-insensitive matching
    3. Splits on whitespace to create individual tokens

    Args:
        document: The text document to tokenise.

    Returns:
        A list of lowercase tokens (words) from the document.

    Example:
        >>> tokenise_document("Hello, World! It's a test.")
        ['hello', 'world', "it's", 'a', 'test']
    """
    clean_doc = re.sub(PUNCTUATION_REGEX, "", document).lower()
    return clean_doc.split()


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
            - DOUBLE_NORMALISATION_K: k + (1 - k) * (count / max_count)
              where k=0.5, preventing bias towards longer documents

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
            return 1.0 if term in tokenised_document else 0.0
        case WeightingSchemes.RAW_COUNT:
            return float(raw_count)
        case WeightingSchemes.TERM_FREQUENCY:
            return raw_count / len(tokenised_document)
        case WeightingSchemes.LOG_NORMALISATION:
            return math.log(1 + raw_count)
        case WeightingSchemes.DOUBLE_NORMALISATION_K:
            # Calculate maximum term frequency in the document
            counts: dict[str, int] = {}
            for token in tokenised_document:
                counts[token] = counts.get(token, 0) + 1
            maximum_count = max(counts.values()) if counts else 0

            if maximum_count == 0:
                return 0.0

            k = 0.5
            return k + ((1 - k) * raw_count) / maximum_count


def compute_inverse_document_frequency(term: str, tokenised_corpus: list[list[str]]) -> float:
    """
    Compute inverse document frequency for a term across a corpus.

    IDF measures how rare or common a term is across all documents. Terms that appear
    in many documents have lower IDF scores, while rare terms have higher scores.

    Formula: `log((1 + N) / (1 + n_t))`
             where N = total documents, n_t = documents containing term

    The +1 smoothing prevents division by zero and reduces the impact of terms
    that appear in all documents.

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
    # TODO: Decide whether to adjust numerator and denominator to avoid div by 0 errors
    term = term.lower()
    num_docs_with_term = 0
    for tokenised_document in tokenised_corpus:
        if term in tokenised_document:
            num_docs_with_term += 1

    return math.log((1 + len(tokenised_corpus)) / (1 + num_docs_with_term))


def compute_tfidf(
    term: str, document: str, corpus: list[str], tf_weighting_scheme: WeightingSchemes = WeightingSchemes.TERM_FREQUENCY
) -> float:
    """
    Compute TF-IDF score for a term in a document within a corpus.

    TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that
    reflects how important a word is to a document in a collection of documents.
    It increases proportionally with the number of times a word appears in the document
    but is offset by the frequency of the word in the corpus.

    Formula: `TF-IDF = TF(term, document) * IDF(term, corpus)`
             where TF = Term Frequency and IDF = Inverse Document Frequency

    Args:
        term: The term to compute TF-IDF for (case-insensitive).
        document: The document text (will be tokenised).
        corpus: A list of all document texts in the corpus (including the target document).
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
    """
    tokenised_document = tokenise_document(document)
    tokenised_corpus = [tokenise_document(doc) for doc in corpus]
    tf = compute_term_frequency(term, tokenised_document, tf_weighting_scheme)
    idf = compute_inverse_document_frequency(term, tokenised_corpus)

    return tf * idf
