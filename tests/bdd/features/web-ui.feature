@web-ui @playwright @headless-browser
Feature: llama-server Built-In Web User Interface
  As a machine learning developer or operator
  I want to interact with the llama-server WebUI using a headless browser
  So that I can verify model availability, inspect server status, and execute chat completions interactively

  Background:
    Given a running llama-server serving model library "<lib_dir>" on "<host>:<port>"

  @smoke @ui-load
  Scenario: Web UI loads successfully in headless browser
    When I navigate to "http://<host>:<port>/" using a headless browser
    Then the browser page title or header should display the application brand
    And the chat conversation container should be visible

  @model-info @verification @known-issue @meta-100
  Scenario: Active model name is displayed in the Web UI
    When I navigate to "http://<host>:<port>/" using a headless browser
    Then the server properties should be loaded from "/props"
    And the UI should display the active model name matching the library manifest

  @chat-completion @e2e
  Scenario: Submit chat prompt and receive streaming completion in browser
    When I navigate to "http://<host>:<port>/" using a headless browser
    And I enter prompt "Hello world" into the chat textarea
    And I click the send message button
    Then an assistant response message should appear in the conversation
    And the assistant message content should be non-empty

  @settings @configuration
  Scenario: Open settings dialog and verify configuration controls
    When I navigate to "http://<host>:<port>/" using a headless browser
    And I open the settings dialog
    Then the settings modal dialog should be visible
    And sampling controls including temperature should be configurable
