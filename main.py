import argparse
import re
import zipfile

import pandas as pd
import torch
from databench_eval import Evaluator, Runner, utils
from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        local_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model)
    results = []
    for p in tqdm(prompts, total=len(prompts)):
        inputs = tokenizer(p, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.4,
            top_p=0.5,
            top_k=10,
            num_beams=4,
            pad_token_id=tokenizer.eos_token_id,
            return_legacy_cache=True,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
                early_stopping=True,
            )
        seq = generation_output.sequences[0]
        output = tokenizer.decode(seq, skip_special_tokens=True)
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
            Ensure the function retains the original column names of the dataset.
        
        import pandas as pd
        def answer(df: pd.DataFrame) -> None:
            df.columns = {list(df.columns)}  # Retain original column names
            # Your solution goes here
            ... 
            >>>{question}
        """


def example_postprocess(response: str, dataset: str, loader):
    matches = re.search(r"(def answer\(df:(.*\n)*)", response)
    try:
        df = loader(dataset)
        exec(
            f"""
global ans
{matches.group(1)}
ans = answer(df)
            """
        )
        # no true result is > 1 line atm, needs 1 line for txt format
        result = ans.split("\n")[0] if "\n" in str(ans) else ans
        return (response, result)
    except Exception as e:
        return (response, f"__CODE_ERROR__: {e}")


def main():
    qa = utils.load_qa(name="semeval", split="dev")
    # qa = Dataset.from_pandas(pd.DataFrame(qa).head())
    evaluator = Evaluator(qa=qa)
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
        resp_eval = []
        with (
            open("predictions.txt", "w", encoding="utf-8") as f1,
            open("debug.txt", "w", encoding="utf-8") as f2,
        ):
            for code, response in responses:
                f1.write(str(response) + "\n")
                if debug:
                    f2.write(f"{code}\nResponse: {str(response)}\n{'-'*20}\n")
                resp_eval.append(str(response))
        print(f"DataBench accuracy is {evaluator.eval(resp_eval)}")  # ~0.16

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
        resp_eval = []
        with (
            open("predictions_lite.txt", "w", encoding="utf-8") as f1,
            open("debug_lite.txt", "w", encoding="utf-8") as f2,
        ):
            for code, response in responses_lite:
                f1.write(str(response) + "\n")
                if debug:
                    f2.write(f"{code}\nResponse: {str(response)}\n{'-'*20}\n")
                resp_eval.append(str(response))
        print(
            f"DataBench_lite accuracy is {evaluator.eval(resp_eval, lite=True)}"
        )  # ~0.08

    if zip_file:
        with zipfile.ZipFile("submission.zip", "w") as zipf:
            if task in ["task-1", "all"]:
                zipf.write("predictions.txt")
            if task in ["task-2", "all"]:
                zipf.write("predictions_lite.txt")

        print(
            "Created submission.zip containing predictions.txt and predictions_lite.txt"
        )


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

    main()
