from src.llm.azure_openai import parse_llm_plan


def test_parse_valid_simple():
    assert parse_llm_plan('["research", "analysis"]') == ["research", "analysis"]


def test_parse_case_and_whitespace():
    assert parse_llm_plan('  [  "Research" ,  "ANALYSIS" ] ') == ["research", "analysis"]


def test_parse_with_fence_and_text():
    text = """
    Here is the plan:
    ```json
    ["memory", "research", "analysis"]
    ```
    Execute accordingly.
    """
    assert parse_llm_plan(text) == ["memory", "research", "analysis"]


def test_parse_dedup_and_limit():
    assert parse_llm_plan('["memory", "research", "research", "analysis", "extra"]') == ["memory", "research", "analysis"]


def test_parse_invalid_entries():
    assert parse_llm_plan('["foo", 123, {}, "analysis"]') == ["analysis"]


def test_parse_malformed_json():
    assert parse_llm_plan('not json [broken') == []


def test_parse_no_array_found():
    assert parse_llm_plan('No list here, just text') == []
