import argparse
from typing import List

from databench_eval.utils import load_sample
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from databench_eval import utils
import pandas as pd

from datasets import Dataset


class ColumnsSelector:
    def __init__(self, model_name: str):
        self.model = ChatGroq(model_name=model_name)
        self.system_message_content = """
        You are a tabular QA system specialized in understanding and analyzing datasets. Your task is to identify the most relevant columns from a given dataset that can answer a specific question.

        You will be provided with a list of column names from the dataset.
        Based on the question, analyze the provided column names and determine which ones are likely to contain the information required to answer the question. You only have to answer the question based on the provided column names in the formmating described below.
       
        Input Format:
            column_names: A list of column names from the dataset.
            question: A string containing the question to be answered.

        Output Format:
            A list of the relevant column names. The output should be a subset of the provided column names.If no columns are relevant, return an empty list. 
            Only the relevant column names should be returned in list format, without any additional information or formatting.
            
        Example:
            column_names: ["Name", "Age", "Email", "Purchase Date", "Product"]
            question: "Which product was purchased?"
            Output: ["Product"]
        
        Input:
            column_names: {column_names}
            question: {question}

        Output:
        """

    def process(self, question: str, column_names: List[str], dataset: pd.DataFrame):
        
        prompt = self.system_message_content.format(column_names=column_names,question=question)

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Here are 5 rows of the dataset as an example:\n"+str(dataset))
        ]

        return self.model.invoke(messages).content


def process_row(row, column_selector):
    question = row["question"]
    dataset = utils.load_table(row["dataset"])

    # print(f"Processing a question {question} with this column names {list(dataset.columns)}\n\n")
    return column_selector.process(question, list(dataset.columns), dataset.head())


def main():
    qa = utils.load_qa(name="semeval", split="dev")
    qa = pd.DataFrame(qa)
    qa = qa.head()
    
    column_selector = ColumnsSelector(model_name=args.groq_model)

    qa["selected_columns"] = qa.apply(
            lambda row: process_row(row, column_selector), axis=1
        )    

    # for row in qa.itertuples():
    #     # print(row)
    #     # print("QUESTION ***************")
    #     # print(row.question)
    #     # print(row.dataset)
    #     df=utils.load_table(row.dataset)
    #     column_selector.process(row.question, list(df.columns))
    
    print(qa.head())
        
    

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(
        prog="Column-Selector", description="Select the most relevant columns from a dataset to answer a specific question."
    )
  
    parser.add_argument(
        "-gm",
        "--groq-model",
        choices=[
            "gemma2-9b-it",
            "gemma-7b-it",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "llama-guard-3-8b",
            "mixtral-8x7b-32768",
            "llama3-70b-8192",
        ],
        help="Choose witch model will be used from Groq API",
        default="llama3-70b-8192",
        nargs="?",
    )


    # Parse arguments
    args = parser.parse_args()

    main()
    