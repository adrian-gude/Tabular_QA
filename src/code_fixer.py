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
                    Role and Context
                    You are a Python-powered Tabular Data Question-Answering System. Your core expertise lies in understanding tabular datasets and crafting Python scripts to generate precise solutions to user queries.

                    Task Description:
                    Fix the Python code to address a query based on the provided dataset. The output must:

                    - Use the dataset and query as given, avoiding any external assumptions.
                    - Adhere to strict syntax rules for Python, ensuring the code runs flawlessly without external modifications.
                    - Retain the original column names of the dataset in your script.
                    
                    Input Specification
                        code: The Python code that needs to be fixed.
                        error: The error message that results from running the code.
                    
                    Output Specification
                        Return only the Python code that solves the query in the function, excluding any introductory explanations or comments. The function must:
                            Include all essential imports.
                            Be concise and functional, ensuring the script can be executed without additional modifications.
                            Use the dataset and return a result of type number, categorical value, boolean value, or a list of values.
                    """
                ),
            },
            {
                "role": "user",
                "content": f"""
                            Code:
                                Below is the piece of code that needs to be fixed, along with the error message that results from running the code:
                                {response}
                                
                            Error: {error}
                            """,
            },
        ]

        outputs = self.pipe(messages, max_new_tokens=512, return_full_text=False)
        output = outputs[0]["generated_text"]
        return output


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    cf = CodeFixer(model_name)
    response = """def add(a, b):\n
                return a + b\n
                """
    error = "IndentationError: unindent does not match any outer indentation level."
    fixed_code = cf.code_fix(response, error)
    print(fixed_code)


if __name__ == "__main__":
    main()
