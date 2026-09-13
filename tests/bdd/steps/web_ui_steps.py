"""Step definitions for web-ui.feature using Playwright headless browser."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest
import requests
from pytest_bdd import given, when, then, parsers


def is_server_listening(host: str, port: int) -> bool:
    """Check if a server is reachable on host:port."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, ConnectionRefusedError):
        return False


@given(parsers.parse('a running llama-server serving model library "{lib_dir}" on "{address}"'))
def step_running_server_for_webui(bdd_context, lib_dir, address):
    # Parse host:port
    parts = address.replace("<host>", bdd_context.host).replace("<port>", str(bdd_context.port)).split(":")
    host, port = parts[0], int(parts[1])
    bdd_context.host = host
    bdd_context.port = port
    bdd_context.server_url = f"http://{host}:{port}"

    # Fail closed: must be a real running llama-server process
    if not is_server_listening(host, port):
        pytest.skip(
            f"Prerequisite unmet: llama-server is not running at {bdd_context.server_url}. "
            "Start llama-server with --model-dir to execute browser E2E tests."
        )

    # Verify server responds on /health
    try:
        resp = requests.get(f"{bdd_context.server_url}/health", timeout=2.0)
        if resp.status_code != 200:
            pytest.skip(f"Prerequisite unmet: server at {bdd_context.server_url}/health returned {resp.status_code}")
    except requests.RequestException as exc:
        pytest.skip(f"Prerequisite unmet: server at {bdd_context.server_url} unreachable: {exc}")


@when(parsers.parse('I navigate to "{url}" using a headless browser'))
def step_navigate_headless_browser(bdd_context, browser_context, url):
    resolved_url = bdd_context.resolve_placeholder(url)
    try:
        response = browser_context.goto(resolved_url, wait_until="domcontentloaded", timeout=10000)
        assert response is not None, f"No response from {resolved_url}"
        assert response.status == 200, f"Expected HTTP 200 from {resolved_url}, got {response.status}"
    except Exception as exc:
        pytest.fail(f"Headless browser navigation to {resolved_url} failed: {exc}")
    bdd_context.browser_page = browser_context


@then('the browser page title or header should display the application brand')
def step_verify_browser_brand(bdd_context):
    page = bdd_context.browser_page
    assert page is not None
    page.wait_for_selector("div.text-2xl, title", timeout=5000)
    title = page.title()
    brand_headers = page.locator("div.text-2xl:has-text('ik_llama.cpp'), div.text-2xl:has-text('llama.cpp')")
    has_brand = "ik_llama.cpp" in title or "llama.cpp" in title or brand_headers.count() > 0
    assert has_brand, f"Brand 'ik_llama.cpp' or 'llama.cpp' not found. Title: {title}"


@then('the chat conversation container should be visible')
def step_verify_conversation_container(bdd_context):
    page = bdd_context.browser_page
    assert page is not None
    textarea = page.locator("textarea")
    textarea.wait_for(state="visible", timeout=5000)
    assert textarea.is_visible(), "Chat textarea input is not visible"

    sidebar = page.locator("div.drawer-side, [aria-label='Sidebar']")
    sidebar.wait_for(state="attached", timeout=5000)
    assert sidebar.count() > 0, "Sidebar conversation drawer not attached"


@then('the server properties should be loaded from "/props"')
def step_verify_props_loaded(bdd_context):
    page = bdd_context.browser_page
    assert page is not None
    # Validate props endpoint is reachable from the server directly
    resp = requests.get(f"{bdd_context.server_url}/props", timeout=3.0)
    assert resp.status_code == 200, f"/props returned {resp.status_code}"
    props = resp.json()
    assert "default_generation_settings" in props or "model_name" in props
    bdd_context.server_props = props


@then('the UI should display the active model name matching the library manifest')
def step_verify_model_name_displayed(bdd_context):
    page = bdd_context.browser_page
    assert page is not None
    expected_model = getattr(bdd_context, "server_props", {}).get("model_name")
    if not expected_model and bdd_context.manifest:
        expected_model = bdd_context.manifest.get("source", {}).get("name")
    assert expected_model, "Could not determine expected active model name"

    model_locator = page.locator(f"text={expected_model}")
    model_locator.wait_for(state="visible", timeout=5000)
    assert model_locator.is_visible(), f"Model name '{expected_model}' not displayed in UI"


@when(parsers.parse('I enter prompt "{prompt}" into the chat textarea'))
def step_enter_chat_prompt(bdd_context, prompt):
    page = bdd_context.browser_page
    assert page is not None
    textarea = page.locator("textarea")
    textarea.wait_for(state="visible", timeout=5000)
    textarea.fill(prompt)
    bdd_context.last_prompt = prompt


@when('I click the send message button')
def step_click_send_button(bdd_context):
    page = bdd_context.browser_page
    assert page is not None
    # Click send button if present, or submit via Enter
    send_btn = page.locator("button:has(svg.bi-arrow-up), button[aria-label*='Send']")
    if send_btn.count() > 0 and send_btn.first.is_visible():
        send_btn.first.click()
    else:
        textarea = page.locator("textarea")
        textarea.press("Enter")


@then('an assistant response message should appear in the conversation')
def step_assistant_message_appears(bdd_context):
    page = bdd_context.browser_page
    assert page is not None
    # Wait for an assistant message bubble to render
    assistant_msg = page.locator(
        "[aria-description*='Message from assistant'], div.chat-bubble, div[role='group']:has-text('assistant')"
    ).first
    assistant_msg.wait_for(state="visible", timeout=15000)
    assert assistant_msg.is_visible(), "Assistant response bubble did not appear within timeout"
    bdd_context.assistant_bubble = assistant_msg


@then('the assistant message content should be non-empty')
def step_assistant_message_non_empty(bdd_context):
    bubble = getattr(bdd_context, "assistant_bubble", None)
    assert bubble is not None, "No assistant message bubble captured"
    text = bubble.text_content()
    assert text is not None, "Assistant message text_content is None"
    clean_text = text.strip()
    assert len(clean_text) > 0 and clean_text != "...", f"Assistant response was empty: '{clean_text}'"


@when('I open the settings dialog')
def step_open_settings(bdd_context):
    page = bdd_context.browser_page
    assert page is not None
    settings_btn = page.locator("button:has(svg.bi-gear), [data-tip='Settings'] button").first
    settings_btn.wait_for(state="visible", timeout=5000)
    settings_btn.click()


@then('the settings modal dialog should be visible')
def step_settings_modal_visible(bdd_context):
    page = bdd_context.browser_page
    assert page is not None
    modal = page.locator("dialog[open], div.modal-open").first
    modal.wait_for(state="visible", timeout=5000)
    assert modal.is_visible(), "Settings modal dialog was not opened"
    bdd_context.settings_modal = modal


@then('sampling controls including temperature should be configurable')
def step_sampling_controls_configurable(bdd_context):
    modal = getattr(bdd_context, "settings_modal", None)
    assert modal is not None, "Settings modal not captured"
    temp_ctrl = modal.locator("input[name='temperature'], input[type='range'], label:has-text('temperature'), label:has-text('Temperature')").first
    assert temp_ctrl.count() > 0 and temp_ctrl.is_visible(), "Temperature sampler control not found in settings modal"
