from datasets import load_dataset
import pandas as pd
from typing import List
import ast
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import pandas as pd
from typing import List

semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split="train")

class AnswerToQuestion():
    def __init__(self, model_name:str):
        self.model = ChatGroq(model_name=model_name)
        self.system_message_content = """
        You are a specialized tabular QA system adept at understanding and analyzing datasets to generate Python solutions. Your primary goal is to respond to a specific query using the provided dataset, following the precise formats detailed below.

            Task: Write Python code to answer a given question based on the provided dataset. Use the dataset and question exactly as provided, without introducing additional assumptions.
            Output: Return only Python code that answers the question.
            Constraints: Preserve the original column names from the dataset in your code and adhere strictly to the output and input specifications provided.
            Input Format:

            dataset: A Pandas DataFrame object representing the dataset.
            question: A string containing the question to be answered.
            Output Format:

            Python code in a complete script format that solves the given question using the dataset. The code should be written in Python and should not contain any syntax errors. The code should be able to run without any modifications.
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
    

    model = AnswerToQuestion("mixtral-8x7b-32768")
    model_answer = model.process(sample_qa['question'][0], df.head())
    print(f"MODEL ANSWER:\n{model_answer}")

if __name__ == "__main__":
    main()