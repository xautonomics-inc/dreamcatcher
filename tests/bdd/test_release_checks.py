"""Bind the release-checks feature to pytest-bdd."""

from pytest_bdd import scenarios

scenarios("features/release-checks.feature")
