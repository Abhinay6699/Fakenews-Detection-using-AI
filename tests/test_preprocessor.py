import pytest
from app.preprocessor import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_strip_urls(preprocessor):
    text = "Check this out https://example.com/fake-news and www.google.com"
    expected = "Check this out  and "
    assert preprocessor.strip_urls(text) == expected

def test_lowercase(preprocessor):
    text = "ThIs Is MiXeD CaSe"
    expected = "this is mixed case"
    assert preprocessor.lowercase(text) == expected

def test_remove_punctuation(preprocessor):
    text = "Hello, world! How's it going?"
    expected = "Hello world Hows it going"
    assert preprocessor.remove_punctuation(text) == expected

def test_tokenize(preprocessor):
    text = "this is a test"
    expected = ["this", "is", "a", "test"]
    assert preprocessor.tokenize(text) == expected

def test_remove_stopwords(preprocessor):
    tokens = ["this", "is", "a", "test", "with", "stopwords"]
    expected = ["test", "stopwords"]
    assert preprocessor.remove_stopwords(tokens) == expected

def test_lemmatize(preprocessor):
    tokens = ["running", "dogs", "better"]
    expected = ["running", "dog", "better"] # Note: default pos is noun, so running stays
    assert preprocessor.lemmatize(tokens) == expected

def test_full_pipeline(preprocessor):
    text = "BREAKING: See this link https://t.co/xyz!! The dogs are running fast."
    expected = "breaking see link dog running fast"
    assert preprocessor.preprocess(text) == expected
