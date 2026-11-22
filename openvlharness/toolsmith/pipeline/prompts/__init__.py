"""Prompt templates of the tool-generation pipeline, as listed in the paper's appendix."""
from functools import cache
from importlib.resources import files


@cache
def prompt(name):
    """Return the text of prompts/<name>.txt."""
    return files(__name__).joinpath(f"{name}.txt").read_text()
