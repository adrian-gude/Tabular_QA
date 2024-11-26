import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

## MODEL CONFIG
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
