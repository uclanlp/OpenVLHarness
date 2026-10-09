"""Judge model clients. ``HFChatModel`` (local transformers models) is imported lazily so
that API judges never pull in torch."""
from .gpt import OpenAIWrapper, GPT4V

__all__ = ['OpenAIWrapper', 'HFChatModel', 'GPT4V']


def __getattr__(name):
    if name == 'HFChatModel':
        from .hf_chat_model import HFChatModel
        return HFChatModel
    raise AttributeError(name)
