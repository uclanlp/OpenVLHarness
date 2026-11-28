"""Prompt templates of the tool-generation pipeline, adapted from the paper's appendix to this codebase's tool interface."""
from functools import cache
from importlib.resources import files


@cache
def prompt(name):
    """Return the text of prompts/<name>.txt."""
    return files(__name__).joinpath(f"{name}.txt").read_text()
