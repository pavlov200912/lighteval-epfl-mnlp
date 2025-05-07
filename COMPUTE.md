# CS 552 Modern NLP Project Compute Tutorial

This year, we will be using EPFL's RCP educational cluster section as the main compute for students. 
You will be operating on the cluster via the [Gnoto](https://gnoto.epfl.ch/) JupyterLab environment.

## Step 1: Login to Gnoto:
Proceed with the link and login with your **Tequila** credential.

## Step 3: Create Virtual Environment

my_venvs_create mnlp_m2
my_venvs_activate mnlp_m2

## Step 4: Install Default Packages

pip install torch
pip install transformers

## A simple model loading file

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```

## Install evaluation suite


## Setup evaluation suite


## Create 



