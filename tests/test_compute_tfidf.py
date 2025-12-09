"""
Test suite for TF-IDF score calculation.

Based on the example from https://medium.com/nlplanet/two-minutes-nlp-learn-tf-idf-with-easy-examples-7c15957b4cb3:
Q: The cat.
D1: The cat is on the mat.
D2: My dog and cat are the best.
D3: The locals are playing.

"""

import math

import pytest

from src.tfidf import WeightingSchemes, compute_inverse_document_frequency, compute_term_frequency, compute_tfidf, tokenise_document


class TestTFIDFBasicExample:
    """Test TF-IDF calculation using the basic example."""

    @pytest.fixture
    def corpus(self) -> list[str]:
        """The corpus from the example."""
        return ["The cat is on the mat.", "My dog and cat are the best.", "The locals are playing."]

    @pytest.fixture
    def tokenised_corpus(self, corpus: list[str]) -> list[list[str]]:
        """Tokenised version of the corpus."""
        return [tokenise_document(doc) for doc in corpus]

    def test_term_frequency_the_in_d1(self, corpus: list[str]) -> None:
        """Test TF('the', D1) = 2/6 = 0.33"""
        document = corpus[0]
        tokenised_doc = tokenise_document(document)
        tf = compute_term_frequency("the", tokenised_doc, WeightingSchemes.TERM_FREQUENCY)
        expected = 2 / 6
        assert pytest.approx(tf, rel=1e-2) == expected

    def test_term_frequency_the_in_d2(self, corpus: list[str]) -> None:
        """Test TF('the', D2) = 1/7 = 0.14"""
        document = corpus[1]
        tokenised_doc = tokenise_document(document)
        tf = compute_term_frequency("the", tokenised_doc, WeightingSchemes.TERM_FREQUENCY)
        expected = 1 / 7
        assert pytest.approx(tf, rel=1e-2) == expected

    def test_term_frequency_the_in_d3(self, corpus: list[str]) -> None:
        """Test TF('the', D3) = 1/4 = 0.25"""
        document = corpus[2]
        tokenised_doc = tokenise_document(document)
        tf = compute_term_frequency("the", tokenised_doc, WeightingSchemes.TERM_FREQUENCY)
        expected = 1 / 4
        assert pytest.approx(tf, rel=1e-2) == expected

    def test_term_frequency_cat_in_d1(self, corpus: list[str]) -> None:
        """Test TF('cat', D1) = 1/6 = 0.17"""
        document = corpus[0]
        tokenised_doc = tokenise_document(document)
        tf = compute_term_frequency("cat", tokenised_doc, WeightingSchemes.TERM_FREQUENCY)
        expected = 1 / 6
        assert pytest.approx(tf, rel=1e-2) == expected

    def test_term_frequency_cat_in_d2(self, corpus: list[str]) -> None:
        """Test TF('cat', D2) = 1/7 = 0.14"""
        document = corpus[1]
        tokenised_doc = tokenise_document(document)
        tf = compute_term_frequency("cat", tokenised_doc, WeightingSchemes.TERM_FREQUENCY)
        expected = 1 / 7
        assert pytest.approx(tf, rel=1e-2) == expected

    def test_term_frequency_cat_in_d3(self, corpus: list[str]) -> None:
        """Test TF('cat', D3) = 0/4 = 0"""
        document = corpus[2]
        tokenised_doc = tokenise_document(document)
        tf = compute_term_frequency("cat", tokenised_doc, WeightingSchemes.TERM_FREQUENCY)
        expected = 0
        assert tf == expected

    def test_inverse_document_frequency_the(self, tokenised_corpus: list[list[str]]) -> None:
        """Test IDF('the') = log(3/3) = log(1) = 0"""
        idf = compute_inverse_document_frequency("the", tokenised_corpus)
        # Since "the" appears in all 3 documents: log((1+3)/(1+3)) = log(1) = 0
        expected = math.log(1)
        assert pytest.approx(idf, abs=0.01) == expected

    def test_inverse_document_frequency_cat(self, tokenised_corpus: list[list[str]]) -> None:
        """Test IDF('cat') = log(3/2) = 0.18"""
        idf = compute_inverse_document_frequency("cat", tokenised_corpus)
        # "cat" appears in 2 out of 3 documents: log((1+3)/(1+2)) = log(4/3)
        expected = math.log(4 / 3)
        assert pytest.approx(idf, rel=1e-2) == expected


class TestTFIDFScoreCalculation:
    """Test complete TF-IDF score calculations."""

    @pytest.fixture
    def corpus(self) -> list[str]:
        """The corpus from the example."""
        return ["The cat is on the mat.", "My dog and cat are the best.", "The locals are playing."]

    def test_tfidf_the_in_d1(self, corpus: list[str]) -> None:
        """Test TF-IDF('the', D1) = 0.33 * 0 = 0"""
        score = compute_tfidf("the", corpus[0], corpus, WeightingSchemes.TERM_FREQUENCY)
        # TF = 2/6 = 0.33, IDF = log(1) = 0, so TF-IDF = 0
        assert pytest.approx(score, abs=0.01) == 0

    def test_tfidf_the_in_d2(self, corpus: list[str]) -> None:
        """Test TF-IDF('the', D2) = 0.14 * 0 = 0"""
        score = compute_tfidf("the", corpus[1], corpus, WeightingSchemes.TERM_FREQUENCY)
        # TF = 1/7 = 0.14, IDF = log(1) = 0, so TF-IDF = 0
        assert pytest.approx(score, abs=0.01) == 0

    def test_tfidf_the_in_d3(self, corpus: list[str]) -> None:
        """Test TF-IDF('the', D3) = 0.25 * 0 = 0"""
        score = compute_tfidf("the", corpus[2], corpus, WeightingSchemes.TERM_FREQUENCY)
        # TF = 1/4 = 0.25, IDF = log(1) = 0, so TF-IDF = 0
        assert pytest.approx(score, abs=0.01) == 0

    def test_tfidf_cat_in_d1(self, corpus: list[str]) -> None:
        """Test TF-IDF('cat', D1) = 0.17 * 0.29 = 0.0493"""
        score = compute_tfidf("cat", corpus[0], corpus, WeightingSchemes.TERM_FREQUENCY)
        # TF = 1/6 = 0.167, IDF = log(4/3) = 0.288, so TF-IDF ≈ 0.048
        expected = (1 / 6) * math.log(4 / 3)
        assert pytest.approx(score, rel=1e-2) == expected

    def test_tfidf_cat_in_d2(self, corpus: list[str]) -> None:
        """Test TF-IDF('cat', D2) = 0.14 * 0.29 = 0.0406"""
        score = compute_tfidf("cat", corpus[1], corpus, WeightingSchemes.TERM_FREQUENCY)
        # TF = 1/7 = 0.143, IDF = log(4/3) = 0.288, so TF-IDF ≈ 0.041
        expected = (1 / 7) * math.log(4 / 3)
        assert pytest.approx(score, rel=1e-2) == expected

    def test_tfidf_cat_in_d3(self, corpus: list[str]) -> None:
        """Test TF-IDF('cat', D3) = 0 * 0.29 = 0"""
        score = compute_tfidf("cat", corpus[2], corpus, WeightingSchemes.TERM_FREQUENCY)
        # "cat" doesn't appear in D3, so TF = 0, TF-IDF = 0
        assert score == 0

    def test_tfidf_nonexistent_term(self, corpus: list[str]) -> None:
        """Test TF-IDF for a term that doesn't exist in any document."""
        score = compute_tfidf("nonexistent", corpus[0], corpus, WeightingSchemes.TERM_FREQUENCY)
        # Term doesn't appear in document, so TF = 0, TF-IDF = 0
        assert score == 0


class TestTFIDFRanking:
    """Test document ranking based on TF-IDF scores."""

    @pytest.fixture
    def corpus(self) -> list[str]:
        """The corpus from the example."""
        return ["The cat is on the mat.", "My dog and cat are the best.", "The locals are playing."]

    def test_cat_relevance_ranking(self, corpus: list[str]) -> None:
        """Test that 'cat' has higher relevance in D1 and D2 than D3."""
        score_d1 = compute_tfidf("cat", corpus[0], corpus, WeightingSchemes.TERM_FREQUENCY)
        score_d2 = compute_tfidf("cat", corpus[1], corpus, WeightingSchemes.TERM_FREQUENCY)
        score_d3 = compute_tfidf("cat", corpus[2], corpus, WeightingSchemes.TERM_FREQUENCY)

        # D3 should have score 0 since "cat" doesn't appear
        assert score_d3 == 0
        # D1 and D2 should have positive scores
        assert score_d1 > 0
        assert score_d2 > 0
        # Both D1 and D2 contain "cat", so both should rank higher than D3
        assert score_d1 > score_d3
        assert score_d2 > score_d3

    def test_average_tfidf_for_query(self, corpus: list[str]) -> None:
        """Test average TF-IDF scores for query 'The cat'.

        Expected ranking from the example:
        D1: The cat is on the mat. (avg = 0.048/2 = 0.024)
        D2: My dog and cat are the best. (avg = 0.041/2 = 0.0205)
        D3: The locals are playing. (avg = 0/2 = 0)
        """
        query_terms = ["the", "cat"]

        # Calculate average TF-IDF for each document
        avg_scores = []
        for document in corpus:
            scores = [compute_tfidf(term, document, corpus, WeightingSchemes.TERM_FREQUENCY) for term in query_terms]
            avg_score = sum(scores) / len(scores)
            avg_scores.append(avg_score)

        # D1 should have the highest average score (contains "cat")
        # D2 should have the second highest (contains "cat" but different context)
        # D3 should have the lowest (doesn't contain "cat")
        assert avg_scores[0] >= avg_scores[1] >= avg_scores[2]
        assert avg_scores[2] == 0  # D3 doesn't contain "cat"

    def test_common_term_has_low_impact(self, corpus: list[str]) -> None:
        """Test that common terms like 'the' have low TF-IDF scores."""
        score_the_d1 = compute_tfidf("the", corpus[0], corpus, WeightingSchemes.TERM_FREQUENCY)
        score_cat_d1 = compute_tfidf("cat", corpus[0], corpus, WeightingSchemes.TERM_FREQUENCY)

        # "the" appears in all documents, so IDF = 0, TF-IDF = 0
        assert score_the_d1 == 0
        # "cat" is more distinctive than "the", so should have higher score
        assert score_cat_d1 > score_the_d1


class TestIDFEdgeCases:
    """Test handling of edge cases like empty inputs."""

    @pytest.fixture
    def corpus(self) -> list[str]:
        """The corpus from the example."""
        return ["The cat is on the mat.", "My dog and cat are the best.", "The locals are playing."]

    def test_empty_term(self, corpus: list[str]) -> None:
        """Empty terms should return 0.0 for all schemes."""
        for scheme in WeightingSchemes:
            result = compute_tfidf("", corpus[0], corpus, scheme)
            assert pytest.approx(result, abs=1e-6) == 0.0, f"{scheme} failed on empty term"

    def test_empty_document(self, corpus: list[str]) -> None:
        """Empty documents should return 0.0 for all schemes."""
        for scheme in WeightingSchemes:
            result = compute_tfidf("test", "", corpus, scheme)
            assert result == 0.0, f"{scheme} failed on empty document"

    def test_nonexistent_term_returns_zero(self) -> None:
        """Non-existent terms should return 0.0 for all schemes."""
        doc = ["the", "cat", "sat", "on", "the", "mat"]
        for scheme in WeightingSchemes:
            result = compute_term_frequency("nonexistent", doc, scheme)
            assert result == 0.0, f"{scheme} should return 0.0 for missing term"
