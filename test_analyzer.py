from analyzer import clean_reviews, extract_report, parse_classification


def test_clean_removes_empty_and_duplicates():
    assert clean_reviews(["  Good  ", "", "good", "Bad"]) == ["Good", "Bad"]


def test_parse_valid_json_is_normalized():
    out = parse_classification('{"sentiment": "Positive", "topics": ["Delivery"]}')
    assert out == {"sentiment": "positive", "topics": ["delivery"]}


def test_parse_malformed_output_falls_back():
    assert parse_classification("no json here") == {"sentiment": "unknown", "topics": []}


def test_extract_report_uses_tags_or_falls_back():
    assert extract_report("<report>Summary</report>") == "Summary"
    assert extract_report("plain text") == "plain text"