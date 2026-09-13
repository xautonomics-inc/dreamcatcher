"""Execute BDD scenarios for single-process layer library serving."""

from pytest_bdd import scenarios

scenarios("features/serve-library.feature")
