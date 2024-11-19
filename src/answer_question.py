from datasets import load_dataset
import pandas as pd
from typing import List
import ast
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import pandas as pd
from typing import List

semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split="train")

class AnswerGenerator():
    def __init__(self, model_name:str):
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
            Provide a Python script in a standalone format that:
                Directly solves the query using the dataset.
                Includes essential imports for execution.
                Avoids extraneous code, ensuring the script is concise and functional.
            The return of the function should be either a number, a categorical value, a boolean value or lists of several types.
        
        Code Template
        Below is a reusable code structure for reference:

        import pandas as pd

        def answer(df: pd.DataFrame) -> {expected_return_type}:
            df.columns = {list(df.columns)}  # Retain original column names
            # Your solution goes here
            ... 
        """

    def process(self,question:str, dataset:pd.DataFrame):
        messages = [
            SystemMessage(content=self.system_message_content),
            HumanMessage(content=question),
        ]

        print(f"Processing a question {question} with this dataset {dataset}\n\n")

        return self.model.invoke(messages).content
    
def main():
    # args -> whole dataset, specfic dataset, specific question, model
    sample_qa = semeval_train_qa[:10]
    print(sample_qa)

    # load questions
    print(sample_qa['question'])

    # read forbes dataset
    df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{sample_qa['dataset'][0]}/all.parquet")
    

    model = AnswerGenerator("mixtral-8x7b-32768")
    model_answer = model.process(sample_qa['question'][0], df.head())
    print(f"MODEL ANSWER:\n{model_answer}")

if __name__ == "__main__":
    main()