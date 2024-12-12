import argparse
import re
import zipfile
import datetime

import pandas as pd
from databench_eval import Evaluator, Runner, utils
from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from src.code_fixer import CodeFixer


def call_model_groq(prompts):
    results = []
    model = ChatGroq(model_name=groq_model)
    for p in tqdm(prompts, total=len(prompts)):
        content, question = p.split(">>>")
        messages = [
            SystemMessage(content=content),
            HumanMessage(content=question),
        ]
        result = model.invoke(messages).content
        results.append(result)
    return results


def call_model_local(prompts):
    results = []
    for p in tqdm(prompts, total=len(prompts)):
        content, question = p.split(">>>")
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": question},
        ]
        outputs = pipe(messages, max_new_tokens=512, return_full_text=False)
        output = outputs[0]["generated_text"]
        results.append(output)
    return results


def example_generator(row: dict) -> str:
    """IMPORTANT:
    **Only the question and dataset keys will be available during the actual competition**.
    You can, however, try to predict the answer type or columns used
    with another modeling task if that helps, then use them here.
    """
    dataset = row["dataset"]
    question = row["question"]
    df = utils.load_table(dataset)
    columns = row.get("columns_used").strip("][").replace("'", "").split(", ")
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
            # The columns used in the solution : {columns}
            # Your solution goes here
            ... 
            >>>{question}
        """


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
        except Exception as e:
            return (f"{response}\n{response_fixed}", f"__CODE_ERROR__: {e}")


def main():
    qa = utils.load_qa(name="semeval", split="dev")
    qa = Dataset.from_pandas(pd.DataFrame(qa))
    evaluator = Evaluator(qa=qa)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if task in ["task-1", "all"]:
        runner = Runner(
            model_call=call_model_groq if mode == "groq" else call_model_local,
            prompt_generator=example_generator,
            postprocess=lambda response, dataset: example_postprocess(
                response, dataset, utils.load_table
            ),
            qa=qa,
            batch_size=10,
        )
        responses = runner.run()
        resp_eval = [str(response) for _, response in responses]
        accuracy = evaluator.eval(resp_eval)
        print(f"DataBench accuracy is {accuracy}")  # ~0.16
        with (
            open("predictions.txt", "w", encoding="utf-8") as f1,
            open(f"{date}_debug.txt", "w", encoding="utf-8") as f2,
        ):
            if debug:
                f2.write(f"Model:{local_model}\nAccuracy:{accuracy}\n{'-'*10}\n")
            for code, response in responses:
                f1.write(str(response) + "\n")
                if debug:
                    f2.write(f"{code}\nResponse: {str(response)}\n{'-'*20}\n")
        print("Created predictions.txt")

    if task in ["task-2", "all"]:
        runner_lite = Runner(
            model_call=call_model_groq if mode == "groq" else call_model_local,
            prompt_generator=example_generator,
            postprocess=lambda response, dataset: example_postprocess(
                response, dataset, utils.load_sample
            ),
            qa=qa,
            batch_size=10,
        )
        responses_lite = runner_lite.run()
        resp_eval = [str(response) for _, response in responses_lite]
        accuracy_lite = evaluator.eval(resp_eval, lite=True)
        print(f"DataBench_lite accuracy is {accuracy_lite}")  # ~0.08
        with (
            open("predictions_lite.txt", "w", encoding="utf-8") as f1,
            open(f"{date}_debug_lite.txt", "w", encoding="utf-8") as f2,
        ):
            if debug:
                f2.write(f"Model:{local_model}\nAccuracy:{accuracy_lite}\n{'-'*10}\n")
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
        "--mode",
        choices=["groq", "local"],
        help="Choose between local execution or Groq API",
        required=True,
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
    parser.add_argument(
        "-lm",
        "--local-model",
        help="Choose witch model will be used from HuggingFace",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        nargs="?",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=["task-1", "task-2", "all"],
        help="Choose witch task will be executed",
        default="all",
        nargs="?",
    )
    parser.add_argument("-z", "--zip-file", action="store_true", help="Create zip file")
    parser.add_argument("--debug", action="store_true", help="Create debug file")

    # Parse arguments
    args = parser.parse_args()
    mode = args.mode
    groq_model = args.groq_model
    local_model = args.local_model
    task = args.task
    zip_file = args.zip_file
    debug = args.debug

    if local_model:
        tokenizer = AutoTokenizer.from_pretrained(local_model)
        model = AutoModelForCausalLM.from_pretrained(
            local_model,
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

    main()
