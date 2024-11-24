from datasets import load_dataset
from databench_eval.utils import load_table
import pandas as pd
from typing import List
import ast
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import pandas as pd
from typing import List


class AnswerGenerator:
    def __init__(self, model_name: str):
        self.model = ChatGroq(model_name=model_name)
        self.system_message_content = """
        Role and Context
        You are a Python-powered Tabular Data Question-Answering System. Your core expertise lies in understanding tabular datasets and crafting Python scripts to generate precise solutions to user queries.

        Task Description:
        Generate Python code to address a query based on the provided dataset. The output must:

        - Use the dataset and query as given, avoiding any external assumptions.
        - Adhere to strict syntax rules for Python, ensuring the code runs flawlessly without external modifications.
        - Retain the original column names of the dataset in your script.
        
        Input Specification
            dataset: A Pandas DataFrame containing the data to be analyzed.
            question: A string outlining the specific query.
        
        Output Specification
            Return only the Python code that solves the query in the function, excluding any introductory explanations or comments. The function must:
                Include all essential imports.
                Be concise and functional, ensuring the script can be executed without additional modifications.
                Use the dataset and return a result of type number, categorical value, boolean value, or a list of values.

        Code Template
            Below is a reusable code structure for reference:
            Return only the code inside the function, without any outer indentation.
            Complete the function with your solution, ensuring the code is functional and concise.
        
        import pandas as pd
        def answer(df: pd.DataFrame) -> {expected_return_type}:
            df.columns = {list(df.columns)}  # Retain original column names
            # Your solution goes here
            ... 
       
        """

    def process(self, question: str, dataset: pd.DataFrame):
        messages = [
            SystemMessage(content=self.system_message_content),
            HumanMessage(content=question),
        ]

        print(f"Processing a question {question} with this dataset {dataset}\n\n")

        return self.model.invoke(messages).content

    def write_response_to_file(self, response: str, output_path: str):

        with open(output_path, "a") as f:
            f.write(response)
            f.write("-" * 50)
            f.write("\n")
        f.close()


def main():
    semeval_train_qa = load_dataset(
        "cardiffnlp/databench", name="semeval", split="train"
    )

    model = AnswerGenerator("gemma2-9b-it")

    for row in semeval_train_qa:
        df = load_table(row["dataset"])
        model_answer = model.process(row["question"], df.head())
        model.write_response_to_file(model_answer, f"example.txt")


if __name__ == "__main__":
    main()
