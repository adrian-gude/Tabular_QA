import zipfile

import pandas as pd
import torch
from databench_eval import Evaluator, Runner, utils
from datasets import Dataset
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
# TODO: complete the following function in one line. It should give the answer to: How many rows are there in this dataframe? 
def example(df: pd.DataFrame) -> int:
    df.columns=["A"]
    return df.shape[0]

# TODO: complete the following function in one line. It should give the answer to: {question}
def answer(df: pd.DataFrame) -> {row["type"]}:
    df.columns = {list(df.columns)}
    return"""


def example_postprocess(response: str, dataset: str, loader):
    try:
        df = loader(dataset)
        lead = """
def answer(df):
    return """
        exec(
            "global ans\n"
            + lead
            + response.split("return")[2]
            .split("\n")[0]
            .strip()
            .replace("[end of text]", "")
            + f"\nans = answer(df)"
        )
        # no true result is > 1 line atm, needs 1 line for txt format
        return ans.split("\n")[0] if "\n" in str(ans) else ans
    except Exception as e:
        return f"__CODE_ERROR__: {e}"


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

runner_lite = Runner(
    model_call=call_model,
    prompt_generator=example_generator,
    postprocess=lambda response, dataset: example_postprocess(
        response, dataset, utils.load_sample
    ),
    qa=qa,
    batch_size=10,
)

responses = runner.run(save="predictions.txt")
responses_lite = runner_lite.run(save="predictions_lite.txt")
print(f"DataBench accuracy is {evaluator.eval(responses)}")  # ~0.16
print(
    f"DataBench_lite accuracy is {evaluator.eval(responses_lite, lite=True)}"
)  # ~0.08


with zipfile.ZipFile("submission.zip", "w") as zipf:
    zipf.write("predictions.txt")
    zipf.write("predictions_lite.txt")

print("Created submission.zip containing predictions.txt and predictions_lite.txt")
