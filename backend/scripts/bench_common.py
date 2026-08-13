"""Shared fixtures for the benchmark scripts in this directory.

Both `bench_chat.py` (full workflow, whatever provider `.env` points at) and
`bench_models.py` (raw prompts against local Ollama models) measure against the
same question set, so the numbers in METRICS.md stay comparable across runs.
"""
SAMPLE_QUESTIONS = [
    "What are your hours on Saturday?",
    "Do you have vegan options on the menu?",
    "What's the story behind the name Two Owls Tavern?",
    "Can I book a table for 10 people?",
    "What's your most popular dish?",
]
