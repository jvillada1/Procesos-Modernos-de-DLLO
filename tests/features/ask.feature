Feature: Ask endpoint (RAG)
  As a client of the FastAPI app
  I want to ask questions
  So that I get an answer using RAG when context is good,
  or a fallback when the context is weak.

  Scenario: RAG returns a good answer when similarity is high
    Given the API client
    And the semantic search returns high-similarity hits
    And the LLM responds with "A concise Minecraft answer."
    When I POST to "/ask" with question "How do I craft a pickaxe?"
    Then the response status is 200
    And the JSON field "used_rag" is true
    And the JSON field "answer" contains "Minecraft answer"
    And the JSON field "sources" contains "Tools • Crafting"

  Scenario: RAG fallback when similarity is too low
    Given the API client
    And the semantic search returns a low-similarity match
    When I POST to "/ask" with question "Unknown lore?"
    Then the response status is 200
    And the JSON field "used_rag" is false
    And the JSON field "answer" equals ""
    And the JSON field "sources" is an empty list
