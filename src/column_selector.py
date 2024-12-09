from typing import List

from databench_eval.utils import load_sample
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from databench_eval import utils
import pandas as pd

from datasets import Dataset

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    Pipeline,
    BitsAndBytesConfig,
)

import torch

class ColumnSelector:
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe

    def select_relevant_columns(self, column_names: List[str], question: str):
        messages = [
            {
                "role": "system",
                "content": (
                    """
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
                ),
            },
            {
                "role": "user",
                "content": f"column_names: {column_names}\nquestion: {question}",
            },
        ]

        outputs = self.pipe(messages, max_new_tokens=512, return_full_text=False)
        output = outputs[0]["generated_text"]
        return output


def process_row(row, column_selector):
    question = row["question"]
    dataset = utils.load_table(row["dataset"])

    return column_selector.select_relevant_columns(list(dataset.columns), question)


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id,
        quantization_config=BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype="auto",
        device_map="auto",
    )

    column_selector = ColumnSelector(pipe)

    qa = utils.load_qa(name="semeval", split="dev")
    qa = pd.DataFrame(qa)
    qa = qa.head()

    qa["selected_columns"] = qa.apply(
        lambda row: process_row(row, column_selector), axis=1
    )

    print(qa.head())


if __name__ == "__main__":
    main()
