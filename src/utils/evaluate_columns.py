from datasets import load_dataset
import pandas as pd
from typing import List
from src.column_selector import ColumnsSelector
import ast
# Load all QA pairs
#all_qa = load_dataset("cardiffnlp/databench", name="qa", split="train")

# Load SemEval 2025 task 8 Question-Answer splits
semeval_train_qa = load_dataset("cardiffnlp/databench", name="semeval", split="train")
#semeval_dev_qa = load_dataset("cardiffnlp/databench", name="semeval", split="dev")



# def evaluate_model_response_to_columns(model_columns_response:List[str], expected_columns:pd.Series):
#     """
#     Evaluate the model response to the expected columns
#     """

#     return set(model_columns_response) == set(expected_columns)
    
if __name__=="__main__":
    sample_qa = semeval_train_qa[:10]
    print(sample_qa)

    # load question 
    print(sample_qa['question'])

    df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{sample_qa['dataset'][0]}/all.parquet")

    model = ColumnsSelector("llama3-70b-8192")
    model_columns = model.process(sample_qa['question'][0], df.columns.to_list())
    print(model_columns)
    try:
        model_columns = ast.literal_eval(model_columns)
        sample_qa['columns_used'][0] = ast.literal_eval(sample_qa['columns_used'][0])
        print("Parsed Model Columns:", model_columns)
        print("Type:", type(model_columns))
        print("Expected Columns:", sample_qa['columns_used'][0])
        print("Type:", type(sample_qa['columns_used'][0]))
        print("Comparison Result:", set(model_columns) == set(sample_qa['columns_used'][0]))
    except Exception as e:
        print("Error in evaluating the model response:", e)
        model_columns = []
        

    


