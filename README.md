# TF-IDF Algorithm

A Python implementation of the Term Frequency-Inverse Document Frequency (TF-IDF) algorithm.

## What is TF-IDF?

In information retrieval, **tf–idf** (term frequency–inverse document frequency) is a measure of importance of a word to a document in a collection or corpus, adjusted for the fact that some words appear more frequently in general. Like the bag-of-words model, it models a document as a multiset of words, without word order. It is a refinement over the simple bag-of-words model, by allowing the weight of words to depend on the rest of the corpus.

TF-IDF is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.

### Components

- **Term Frequency (TF)**: How often a term appears in a document (normalised by document length)
- **Inverse Document Frequency (IDF)**: How rare a term is across all documents

```
TF-IDF(term, document, corpus) = TF(term, document) * IDF(term, corpus)
```

Read more: [TF-IDF on Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## Installation

```bash
git clone https://github.com/cayopolo/tfidf-algorithm.git
cd tfidf-algorithm
uv sync
```

## Usage

### Basic TF-IDF Computation

```python
from src.tfidf import compute_tfidf, tokenise_document

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

# Compute TF-IDF for a term in a document
score = compute_tfidf("cat", corpus[0], corpus)
print(f"TF-IDF score for 'cat': {score}")
```

### Pre-tokenised Input (for efficiency)

When computing TF-IDF for multiple terms, pre-tokenise the corpus once:

```python
from src.tfidf import compute_tfidf, tokenise_document

corpus = ["the cat sat on the mat", "the dog sat on the log"]
tokenised_corpus = [tokenise_document(doc) for doc in corpus]
tokenised_doc = tokenised_corpus[0]

# Reuse tokenised data for multiple queries
for term in ["cat", "sat", "the"]:
    score = compute_tfidf(term, tokenised_doc, tokenised_corpus)
    print(f"{term}: {score:.4f}")
```

### Weighting Schemes

The implementation supports multiple term frequency weighting schemes:

```python
from src.tfidf import compute_tfidf, WeightingSchemes

corpus = ["the cat sat on the mat", "the dog sat on the log"]

# Binary: 1 if term exists, 0 otherwise
score = compute_tfidf("cat", corpus[0], corpus, tf_weighting_scheme=WeightingSchemes.BINARY)

# Raw count: number of occurrences
score = compute_tfidf("the", corpus[0], corpus, tf_weighting_scheme=WeightingSchemes.RAW_COUNT)

# Term frequency (default): count / document length
score = compute_tfidf("cat", corpus[0], corpus, tf_weighting_scheme=WeightingSchemes.TERM_FREQUENCY)

# Log normalisation: log(1 + count)
score = compute_tfidf("cat", corpus[0], corpus, tf_weighting_scheme=WeightingSchemes.LOG_NORMALISATION)
```

| Scheme | Formula | Use Case |
|--------|---------|----------|
| `BINARY` | 1 if present, else 0 | When term presence matters more than frequency |
| `RAW_COUNT` | count(term) | Simple frequency counting |
| `TERM_FREQUENCY` | count / doc_length | Normalised frequency (default) |
| `LOG_NORMALISATION` | log(1 + count) | Dampens impact of high-frequency terms |

