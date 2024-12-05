

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
)
import torch

class CodeFixer:
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe

    def code_fix(self, response: str, error: str):
        messages = [
            {
                "role": "system",
                "content": (
                    """
                    You are a Tabular Data QA system designed to debug Python functions.
                    Your goal is to fix the code provided below to resolve the given error.
                    Ensure the following constraints are met:
                    - The function's signature (name and parameters) must remain unchanged.
                    - Modify only the content within the function's body to address the error.
                    You will be given the function and the corresponding error message as input.
                    Provide the corrected function code that resolves the issue.
                    """
                ),
            },
            {
                "role": "user",
                "content": f"Response: {response}\nError: {error}",
            },
        ]

        outputs = self.pipe(messages, max_new_tokens=512, return_full_text=False)
        output = outputs[0]["generated_text"]
        return output


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    cf = CodeFixer(model_name)
    response = '''def add(a, b):\n
                return a + b\n
                '''
    error = "IndentationError: unindent does not match any outer indentation level."
    fixed_code = cf.code_fix(response, error)
    print(fixed_code)


if __name__ == "__main__":
    main()