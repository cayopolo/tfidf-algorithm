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

