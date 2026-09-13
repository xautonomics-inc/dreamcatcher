"""Execute BDD scenarios for multi-host stage runner pipeline rings."""

from pytest_bdd import scenarios

scenarios("features/stage-ring.feature")
