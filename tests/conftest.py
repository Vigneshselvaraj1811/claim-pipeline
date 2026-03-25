"""Pytest configuration."""
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "anthropic")
