import pytest

from src.tfidf import tokenise_document


class TestBasicTokenisation:
    """Test basic tokenisation functionality."""

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [
            ("hello world", ["hello", "world"]),
            ("hello", ["hello"]),
            ("", []),
            ("   ", []),
            ("hello    world", ["hello", "world"]),
            ("hello\tworld\nfoo\rbar", ["hello", "world", "foo", "bar"]),
        ],
    )
    def test_tokenisation_basics(self, input_str: str, expected: list[str]) -> None:
        """Test basic tokenisation with various inputs."""
        assert tokenise_document(input_str) == expected


class TestCaseNormalisation:
    """Test case conversion to lowercase."""

    @pytest.mark.parametrize(("input_str", "expected"), [("ABC", ["abc"]), ("abc", ["abc"]), ("AbC", ["abc"])])
    def test_case_variations_parametrised(self, input_str: str, expected: list[str]) -> None:
        """Test various case combinations produce same result."""
        assert tokenise_document(input_str) == expected


class TestPunctuation:
    """Test punctuation removal."""

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [
            ("Hello, world! How are you?", ["hello", "world", "how", "are", "you"]),
            ("hello. world.", ["hello", "world"]),
            ("apple, orange, banana", ["apple", "orange", "banana"]),
            ("hello! world!", ["hello", "world"]),
            ("what? why?", ["what", "why"]),
            ("hello... world!!!???", ["hello", "world"]),
            ("(hello) [world] {foo}", ["hello", "world", "foo"]),
            ("first: second; third", ["first", "second", "third"]),
            ("hello-world test–case", ["helloworld", "testcase"]),
            ("hello, world!", ["hello", "world"]),
            ("test. case?", ["test", "case"]),
            ("a, b, c, d", ["a", "b", "c", "d"]),
        ],
    )
    def test_punctuation_removed(self, input_str: str, expected: list[str]) -> None:
        """Test removal of various punctuation marks."""
        assert tokenise_document(input_str) == expected

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [
            ("don't can't won't shouldn't", ["don't", "can't", "won't", "shouldn't"]),
            ("John's book Mary's pen", ["john's", "book", "mary's", "pen"]),
            ("'90s it's", ["90s", "it's"]),
            ('he said "it\'s fine"', ["he", "said", "it's", "fine"]),
        ],
    )
    def test_apostrophes_preserved(self, input_str: str, expected: list[str]) -> None:
        """Test that apostrophes are preserved in contractions and possessives, but quotes are stripped."""
        assert tokenise_document(input_str) == expected

    def test_quotes_removed(self) -> None:
        """Test removal of quotes (both single and double) while preserving apostrophes in contractions."""
        result = tokenise_document("\"hello\" 'world'")
        assert result == ["hello", "world"]

        # Contractions still work
        result = tokenise_document("don't use 'quotes' here")
        assert result == ["don't", "use", "quotes", "here"]


class TestNumbers:
    """Test handling of numeric characters."""

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [("test123 456abc", ["test123", "456abc"]), ("123 456 789", ["123", "456", "789"]), ("abc123! def456?", ["abc123", "def456"])],
    )
    def test_numbers_preserved(self, input_str: str, expected: list[str]) -> None:
        """Test that numeric characters are preserved."""
        assert tokenise_document(input_str) == expected


class TestUnderscores:
    """Test handling of underscores."""

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [("hello_world test_case", ["hello_world", "test_case"]), ("var_1 _private __dunder__", ["var_1", "_private", "__dunder__"])],
    )
    def test_underscores_preserved(self, input_str: str, expected: list[str]) -> None:
        """Test that underscores are preserved."""
        assert tokenise_document(input_str) == expected


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        ("input_str", "expected"),
        [
            ("!!! ??? ...,", []),
            ("  hello world  ", ["hello", "world"]),
            ("hello, world! how\tare\nyou?", ["hello", "world", "how", "are", "you"]),
            ("hello@world #tag $money", ["helloworld", "tag", "money"]),
            ("' '  '", []),
        ],
    )
    def test_edge_cases(self, input_str: str, expected: list[str]) -> None:
        """Test edge cases and boundary conditions."""
        assert tokenise_document(input_str) == expected

    def test_unicode_characters_not_removed(self) -> None:
        """Test handling of unicode word characters."""
        result = tokenise_document("café naïve")
        assert "café" in result
        assert "naïve" in result


class TestRealWorldExamples:
    """Test with realistic text samples."""

    def test_multiline_text(self) -> None:
        """Test multiline document."""
        text = """Once upon a time,
        there was a hero.
        He was brave."""
        result = tokenise_document(text)
        expected = ["once", "upon", "a", "time", "there", "was", "a", "hero", "he", "was", "brave"]
        assert result == expected

    def test_url_like_text(self) -> None:
        """Test text containing URL-like patterns."""
        # TODO: Decide how to handle dots in URLs
        result = tokenise_document("visit example.com for info")
        assert result == ["visit", "examplecom", "for", "info"]

    def test_email_like_text(self) -> None:
        """Test text containing email-like patterns."""
        # TODO: Decide how to handle @ and dots in emails
        result = tokenise_document("contact user@example.com here")
        assert result == ["contact", "userexamplecom", "here"]

    def test_numbers_with_punctuation(self) -> None:
        """Test numbers with punctuation."""
        # TODO: Decide how to handle dots in numbers
        result = tokenise_document("3.14 is pi, 2.71 is e")
        assert result == ["314", "is", "pi", "271", "is", "e"]

    def test_currency_symbols(self) -> None:
        """Test text with currency symbols."""
        # TODO: Decide how to handle currency symbols
        result = tokenise_document("$100 €50 £75")
        assert result == ["100", "50", "75"]
