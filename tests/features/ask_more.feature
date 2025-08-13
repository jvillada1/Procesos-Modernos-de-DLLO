Feature: Ask-more endpoint (Extended answer)
  As a client of the FastAPI app
  I want to request an extended answer
  So that the agent can enrich a previous reply.

  Scenario: Agent returns enriched answer and executed tools
    Given the API client
    And the ExtendedAnswerAgent returns "Extra crafting tips." with a tool "web.search"
    When I POST to "/ask-more" with answer "Base answer"
    Then the response status is 200
    And the JSON field "answer" equals "Extra crafting tips."
    And the JSON field "tools" contains "web.search"
