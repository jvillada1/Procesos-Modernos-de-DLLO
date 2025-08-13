from pytest_bdd import scenarios, given, when, then, parsers
from fastapi.testclient import TestClient
import proyecto

scenarios("ask_more.feature")

@given("the API client", target_fixture="api_client")
def _api_client():
    return TestClient(proyecto.app)

@given(parsers.parse('the ExtendedAnswerAgent returns "{text}" with a tool "{tool_name}"'))
def mock_extended_agent(monkeypatch, text, tool_name):
    class FakeAgent:
        def __init__(self, answer, api_key):
            self.answer = answer
            self.api_key = api_key
        def search(self):
            return {"answer": text, "tools": [{"name": tool_name}]}
    import extended_answer_agent as real_mod
    monkeypatch.setattr(real_mod, "ExtendedAnswerAgent", FakeAgent)

@when(parsers.parse('I POST to "{path}" with answer "{answer}"'))
def post_answer(api_client, path, answer):
    api_client.response = api_client.post(path, json={"answer": answer})

@then("the response status is 200")
def status_ok(api_client):
    assert api_client.response.status_code == 200, api_client.response.text

@then(parsers.parse('the JSON field "{field}" equals "{value}"'))
def field_equals(api_client, field, value):
    data = api_client.response.json()
    assert data.get(field) == value, data

@then(parsers.parse('the JSON field "{field}" contains "{needle}"'))
def field_contains(api_client, field, needle):
    data = api_client.response.json()
    assert needle in str(data.get(field)), data

