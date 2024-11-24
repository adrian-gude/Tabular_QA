from databench_eval.utils import load_sample
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import pandas as pd
from typing import List


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
        """

    def process(self, question: str, column_names: List[str]):
        messages = [
            SystemMessage(content=self.system_message_content),
            HumanMessage(content=str(column_names) + "\n" + question),
        ]

        print(
            f"Processing a question {question} with this column names {column_names}\n\n"
        )

        return self.model.invoke(messages).content


def main():
    qa = load_sample("001_Forbes")
    columns_selector = ColumnsSelector("llama3-70b-8192")
    model_answer = columns_selector.process(
        question="List the top 4 ranks of female billionaires.",
        column_names=qa.columns.to_list(),
    )
    print(f"MODEL ANSWER:\n{model_answer}")


if __name__ == "__main__":
    main()
