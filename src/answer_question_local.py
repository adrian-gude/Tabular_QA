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
    return""