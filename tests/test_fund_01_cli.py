"""Smoke test: fund_01 CLI argparse + --help work."""
import subprocess

REPO = "/Users/shiro/agents/ai-hedge-fund"


def test_fund_01_help_works():
    result = subprocess.run(
        [f"{REPO}/.venv/bin/python3", "-m", "src.cli.fund_01_indian_options", "--help"],
        capture_output=True, text=True, cwd=REPO, timeout=30,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr[:500]}"
    out = result.stdout.lower()
    assert "--symbol" in out
    assert "--portfolio-inr" in out
    assert "--model" in out
