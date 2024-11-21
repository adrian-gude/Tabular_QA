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

### MODEL CONFIG
checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def call_model(prompts):
    results = []
    model = ChatGroq(model_name="llama3-70b-8192")
    for p in tqdm(prompts, total=len(prompts)):
        content, question = p.split(">>>")
        messages = [
            SystemMessage(content=content),
            HumanMessage(content=question),
        ]
        result = model.invoke(messages).content
        results.append(result)
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
        
        import pandas as pd
        def answer(df: pd.DataFrame) -> None:
            df.columns = {list(df.columns)}  # Retain original column names
            # Your solution goes here
            ... 
            >>>{question}
        """


def example_postprocess(response: str, dataset: str, loader):
    re.match(r"```python()```")
    try:
        df = loader(dataset)
        exec(
            f"""
global ans
{response}
ans = answer(df)
            """
        )
        #         lead = """
        # def answer(df):
        #     return """
        #         exec(
        #             "global ans\n"
        #             + lead
        #             + response.split("return")[2]
        #             .split("\n")[0]
        #             .strip()
        #             .replace("[end of text]", "")
        #             + "\nans = answer(df)"
        #         )
        # no true result is > 1 line atm, needs 1 line for txt format
        return ans.split("\n")[0] if "\n" in str(ans) else ans
    except Exception as e:
        return f"__CODE_ERROR__: {e}"


def main():
    qa = utils.load_qa(name="semeval", split="dev")
    qa = Dataset.from_pandas(pd.DataFrame(qa).head())
    evaluator = Evaluator(qa=qa)

    runner = Runner(
        model_call=call_model,
        prompt_generator=example_generator,
        postprocess=lambda response, dataset: example_postprocess(
            response, dataset, utils.load_table
        ),
        qa=qa,
        batch_size=10,
    )

    # runner_lite = Runner(
    #     model_call=call_model,
    #     prompt_generator=example_generator,
    #     postprocess=lambda response, dataset: example_postprocess(
    #         response, dataset, utils.load_sample
    #     ),
    #     qa=qa,
    #     batch_size=10,
    # )

    responses = runner.run(save="predictions.txt")
    # responses_lite = runner_lite.run(save="predictions_lite.txt")
    print(f"DataBench accuracy is {evaluator.eval(responses)}")  # ~0.16
    # print(
    #     f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}"
    # )  # ~0.08

    # with zipfile.ZipFile("submission.zip", "w") as zipf:
    #     zipf.write("predictions.txt")
    #     zipf.write("predictions_lite.txt")

    print("Created submission.zip containing predictions.txt and predictions_lite.txt")


if __name__ == "__main__":
    main()
