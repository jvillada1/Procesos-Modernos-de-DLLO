from pytest_bdd import scenarios, given, when, then, parsers
from fastapi.testclient import TestClient
import proyecto

scenarios("ask.feature")

@given("the API client", target_fixture="api_client")
def _api_client(monkeypatch):
    return TestClient(proyecto.app)

@then(parsers.re(r'^the JSON field "(?P<field>[^"]+)" equals ""$'))
def field_equals_empty(api_client, field):
    data = api_client.response.json()
    assert data.get(field) == "", data

@given("the semantic search returns high-similarity hits")
def mock_search_high(monkeypatch):
    hits = [
        ({"title": "Tools • Crafting", "text": "Use sticks + planks to craft a wooden pickaxe."}, 0.93),
        ({"title": "Mining Basics", "text": "Pickaxes mine stone and ores."}, 0.74),
    ]
    monkeypatch.setattr(proyecto, "search_similar", lambda q, k=6: hits)

@given(parsers.parse('the LLM responds with "{content}"'))
def mock_llm_post(monkeypatch, content):
    class _Resp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": content}}]}
    def _fake_post(url, headers=None, json=None, timeout=None):
        assert "messages" in json
        assert json["model"]
        return _Resp()
    import requests
    monkeypatch.setattr(requests, "post", _fake_post)

@given("the semantic search returns a low-similarity match")
def mock_search_low(monkeypatch):
    low_hits = [({"title": "Unrelated", "text": "Not helpful"}, 0.05)]
    monkeypatch.setattr(proyecto, "search_similar", lambda q, k=6: low_hits)

@when(parsers.parse('I POST to "{path}" with question "{question}"'))
def post_question(api_client, path, question):
    api_client.response = api_client.post(path, json={"question": question})

@then("the response status is 200")
def status_ok(api_client):
    assert api_client.response.status_code == 200, api_client.response.text

@then(parsers.parse('the JSON field "{field}" is true'))
def field_true(api_client, field):
    data = api_client.response.json()
    assert data.get(field) is True, data

@then(parsers.parse('the JSON field "{field}" is false'))
def field_false(api_client, field):
    data = api_client.response.json()
    assert data.get(field) is False, data

@then(parsers.parse('the JSON field "{field}" contains "{snippet}"'))
def field_contains(api_client, field, snippet):
    data = api_client.response.json()
    assert snippet in str(data.get(field, "")), data

@then(parsers.parse('the JSON field "{field}" equals "{value}"'))
def field_equals(api_client, field, value):
    data = api_client.response.json()
    assert data.get(field) == value, data

@then(parsers.parse('the JSON field "{field}" is an empty list'))
def field_empty_list(api_client, field):
    data = api_client.response.json()
    assert data.get(field) == [], data

