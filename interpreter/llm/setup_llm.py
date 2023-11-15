import os

import litellm

from .convert_to_coding_llm import convert_to_coding_llm
from .setup_openai_coding_llm import setup_openai_coding_llm
from .setup_text_llm import setup_text_llm
from .setup_gpt4free_llm import setup_gpt4free_llm 
from .convert_to_coding_gpt4free_llm import convert_to_coding_gpt4free_llm


def setup_llm(interpreter):
    """
    Takes an Interpreter (which includes a ton of LLM settings),
    returns a Coding LLM (a generator that streams deltas with `message` and `code`).
    """
    # gpt4fre
    gpt4free = True
    if gpt4free:
        text_llm = setup_gpt4free_llm(interpreter)
        coding_llm = convert_to_coding_gpt4free_llm(text_llm, debug_mode=interpreter.debug_mode)
        return coding_llm

    # Detect whether or not it's a function calling LLM
    if interpreter.function_calling_llm == None:
        if not interpreter.local and (
            interpreter.model in litellm.open_ai_chat_completion_models
            or interpreter.model.startswith("azure/")
        ):
            interpreter.function_calling_llm = True
        else:
            interpreter.function_calling_llm = False

    if interpreter.function_calling_llm:
        # Function-calling LLM
        coding_llm = setup_openai_coding_llm(interpreter)
    else:
        # If disable_procedures has not been set manually:
        if interpreter.disable_procedures == None:
            # Disable procedures, which confuses most of these models (except GPT-4V)

            if interpreter.model != "gpt-4-vision-preview":
                interpreter.disable_procedures = True

        # Non-function-calling LLM
        text_llm = setup_text_llm(interpreter)
        coding_llm = convert_to_coding_llm(text_llm, debug_mode=interpreter.debug_mode)

    return coding_llm
