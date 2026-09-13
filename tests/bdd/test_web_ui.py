"""Execute BDD scenarios for llama-server WebUI using Playwright."""

from pytest_bdd import scenarios

scenarios("features/web-ui.feature")
