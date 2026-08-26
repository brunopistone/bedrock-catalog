"""
Pytest configuration and fixtures.

Unit tests stub out boto3 and need no credentials. Integration tests call the
real Bedrock control plane and are skipped unless AWS credentials resolve.
"""

import pytest
from click.testing import CliRunner
from dotenv import load_dotenv

from main import cli

# Load .env file for credentials / region
load_dotenv()

# Region used by the integration tests when the environment sets none.
DEFAULT_TEST_REGION = "us-east-1"

AWS_ENV_VARS = ["AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION"]


@pytest.fixture
def runner():
    """Fixture that provides a Click CliRunner."""
    return CliRunner()


@pytest.fixture
def run_cli(runner):
    """
    Fixture that returns a function to run CLI commands.

    Usage:
        result = run_cli(["info"])
        result = run_cli(["list-models", "--region", "us-east-1"])
    """

    def _run_cli(args, input=None):
        return runner.invoke(cli, args, input=input, catch_exceptions=False)

    return _run_cli


@pytest.fixture
def clean_env(monkeypatch):
    """Remove AWS env vars so tests are not affected by .env or the shell."""
    for var in AWS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def aws_region():
    """Region for integration tests, skipping when credentials are missing."""
    import boto3
    from botocore.exceptions import BotoCoreError

    session = boto3.Session()

    try:
        if session.get_credentials() is None:
            pytest.skip("Integration tests require AWS credentials")
    except BotoCoreError as e:
        pytest.skip(f"Integration tests require AWS credentials ({e})")

    return session.region_name or DEFAULT_TEST_REGION


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
