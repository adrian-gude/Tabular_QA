import argparse
import datetime
import re
import zipfile

import pandas as pd
import torch
from databench_eval import Evaluator, Runner
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from src.code_fixer import CodeFixer
from src.column_selector import ColumnSelector

def load_table(name):
    return pd.read_parquet(
        f"./competition/{name}/all.parquet"
    )


def load_sample(name):
    return pd.read_parquet(
        f"./competition/{name}/sample.parquet"
    )
    

def call_model(prompts):
    results = []
    for p in tqdm(prompts, total=len(prompts), dynamic_ncols=True, position=1):
        content, question = p.split(">>>")
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": question},
        ]
        outputs = pipe(messages, max_new_tokens=2048, return_full_text=False)
        output = outputs[0]["generated_text"]
        results.append(output)
    return results


def _format_prompt(row: dict, df: pd.DataFrame, selected_columns: pd.Index, columns_unique, columns_lists, columns_dicts) -> str:
    """IMPORTANT:
    **Only the question and dataset keys will be available during the actual competition**.
    You can, however, try to predict the answer type or columns used
    with another modeling task if that helps, then use them here.
    """
    
    return f"""
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
        def answer(df: pd.DataFrame) -> None:
            df.columns = {list(df.columns)} # Retain original column names 
            # The columns used in the solution : {selected_columns}
            {columns_unique}
            {columns_lists}
            {columns_dicts}
            # Your solution goes here
            ... 
            >>>{row["question"]}
        """


def example_generator(row: dict) -> str:
    column_selector = ColumnSelector(pipe)
    df = load_table(row["dataset"])
    
    selected_columns = column_selector.select_relevant_columns(df.columns, row["question"])
    #print("Model response of selected cols:" + selected_columns)
    columns_unique = column_selector.extract_unique_column_values(df, selected_columns)
    columns_lists = column_selector.extract_list_column_values(df, selected_columns)
    columns_dicts = column_selector.extract_dict_column_values(df, selected_columns)
    
    return _format_prompt(row, df, selected_columns, columns_unique, columns_lists, columns_dicts)


def example_generator_lite(row: dict) -> str:
    column_selector = ColumnSelector(pipe)
    df = load_sample(row["dataset"])
    
    selected_columns = column_selector.select_relevant_columns(df.columns, row["question"])
    columns_unique = column_selector.extract_unique_column_values(df, selected_columns)
    columns_lists = column_selector.extract_list_column_values(df, selected_columns)
    columns_dicts = column_selector.extract_dict_column_values(df, selected_columns)
    
    return _format_prompt(row, df, selected_columns, columns_unique, columns_lists, columns_dicts)


def extract_answer_code(response_text):
    matches = re.search(r"(def answer\(df:(.*\n)*)\`\`\`", response_text)
    if not matches:
        raise ValueError("No function answer definition found in response.")
    code = matches.group(1)
    return code


def execute_answer_code(code, dataset):
    local_namespace = {}
    exec(code, globals(), local_namespace)
    ans = local_namespace["answer"](dataset)
    result = ans.split("\n")[0] if "\n" in str(ans) else ans
    return result


def example_postprocess(response: str, dataset: str, loader):
    df = loader(dataset)
    try:
        code = extract_answer_code(response)
        result = execute_answer_code(code, df)
        return (response, result)
    except Exception as e:
        code_fixer = CodeFixer(pipe)
        response_fixed = code_fixer.code_fix(response, str(e))
        try:
            fixed_code = extract_answer_code(response_fixed)
            result = execute_answer_code(fixed_code, df)
            return (f"{response}\n{response_fixed}", result)
        except Exception as code_error:
            return (f"{response}\n{response_fixed}", f"__CODE_ERROR__: {code_error}")


def main():
    #qa = load_qa(name="semeval", split="dev")
    qa = Dataset.from_pandas(pd.read_csv("competition/test_qa.csv"))
    if n_rows:
        qa = Dataset.from_pandas(pd.read_csv("competition/test_qa.csv").head(n_rows))
    #evaluator = Evaluator(qa=qa)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if task in ["task-1", "all"]:
        runner = Runner(
            model_call=call_model,
            prompt_generator=example_generator,
            postprocess=lambda response, dataset: example_postprocess(
                response, dataset, load_table
            ),
            qa=qa,
            batch_size=batch_size,
        )
        responses = runner.run()
        resp_eval = [str(response) for _, response in responses]
        #accuracy = evaluator.eval(resp_eval)
        #print(f"DataBench accuracy is {accuracy}")  # ~0.16
        with (
            open("predictions.txt", "w", encoding="utf-8") as f1,
            open(f"{date}_debug.txt", "w", encoding="utf-8") as f2,
        ):
            # if debug:
            #     f2.write(f"Model:{model_name}\nAccuracy:{accuracy}\n{'-'*10}\n")
            for code, response in responses:
                f1.write(str(response) + "\n")
                if debug:
                    f2.write(f"{code}\nResponse: {str(response)}\n{'-'*20}\n")
        print("Created predictions.txt")

    if task in ["task-2", "all"]:
        runner_lite = Runner(
            model_call=call_model,
            prompt_generator=example_generator_lite,
            postprocess=lambda response, dataset: example_postprocess(
                response, dataset, load_sample
            ),
            qa=qa,
            batch_size=batch_size,
        )
        responses_lite = runner_lite.run()
        resp_eval = [str(response) for _, response in responses_lite]
        #accuracy_lite = evaluator.eval(resp_eval, lite=True)
        #print(f"DataBench_lite accuracy is {accuracy_lite}")  # ~0.08
        with (
            open("predictions_lite.txt", "w", encoding="utf-8") as f1,
            open(f"{date}_debug_lite.txt", "w", encoding="utf-8") as f2,
        ):
            # if debug:
            #     f2.write(f"Model:{model_name}\nAccuracy:{accuracy_lite}\n{'-'*10}\n")
            for code, response in responses_lite:
                f1.write(str(response) + "\n")
                if debug:
                    f2.write(f"{code}\nResponse: {str(response)}\n{'-'*20}\n")
        print("Created predictions_lite.txt")

    if zip_file:
        with zipfile.ZipFile("submission.zip", "w") as zipf:
            if task in ["task-1", "all"]:
                zipf.write("predictions.txt")
            if task in ["task-2", "all"]:
                zipf.write("predictions_lite.txt")
        print("Created submission.zip")


if __name__ == "__main__":
    # Init ArgumentParser
    parser = argparse.ArgumentParser(
        prog="Tabular-QA", description="Respond questions from tabular data"
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Choose witch model will be used from HuggingFace. Default Mistral 7B Instruct",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        nargs="?",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=["task-1", "task-2", "all"],
        help="Choose witch task will be executed. Default both task (all)",
        default="all",
        nargs="?",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        nargs="?",
        type=int,
        default=10,
        help="Batch size. Default 10",
    )
    parser.add_argument(
        "-n",
        "--number-rows",
        nargs="?",
        type=int,
        default=None,
        help="Only execute n rows from the Dataset. Default all rows",
    )
    parser.add_argument("-z", "--zip-file", action="store_true", help="Create zip file")
    parser.add_argument("--debug", action="store_true", help="Create debug file")

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model
    task = args.task
    batch_size = args.batch_size
    n_rows = args.number_rows
    zip_file = args.zip_file
    debug = args.debug

    # Verify arguments

    if not batch_size or batch_size < 0:
        print(
            f"Warning: Invalid batch_size value '{batch_size}', changed to default (10)"
        )
        batch_size = 10

    if not n_rows and n_rows is not None:
        print(
            f"Warning: Invalid number_rows value '{n_rows}', parameter will be ignore"
        )

    try:
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
    except OSError as e:
        print(e)
        exit(-1)

    main()
