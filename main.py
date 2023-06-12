# %%
import torch
import pandas as pd
import numpy as np
import transformers
from transformers import DataCollatorWithPadding
from datasets import load_dataset, load_metric, Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig

TEST_PATH = "train.csv"
MODEL_NAME = "sberbank-ai/ruRoberta-large"
DEVICE = "cuda"
BATCH_SIZE = 16
SEED = 421
MODEL_MAX_LENGTH = 150

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenizer.padding_side = 'right'
tokenizer.truncation_side = "right"
tokenizer.model_max_length = MODEL_MAX_LENGTH

# %%
test = load_dataset("csv", data_files={"test": TEST_PATH})["test"]

# %%
id2label = {0: "hu_answer", 1: "ai_answer"}
label2id = {"hu_answer": 0, "ai_answer": 1}
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    # torch_dtype=torch.float16,
    # low_cpu_mem_usage=True
)

config = PeftConfig.from_pretrained("models/lora2")
peft_model = PeftModel.from_pretrained(model, "models/lora2")


# %%
def concat(examples):
    inputs = [
        f"{answer}"
        for context, answer in zip(examples["q_title"], examples["ans_text"])
    ]
    return {"text": inputs}


def tokenize(examples):
    outputs = tokenizer(
        examples["text"],
        # padding="max_length",
        truncation=True,
        max_length=MODEL_MAX_LENGTH,
    )

    # outputs["labels"] = [float(label2id[x]) for x in examples["label"]]
    return outputs


test_dataset = test.map(
    concat,
    batched=True,
).map(tokenize, batched=True, remove_columns=[x for x in test.column_names])


# %%
trainer = transformers.Trainer(
    model=model,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
)

# %%
preds = trainer.predict(test_dataset)

# %%
final_preds = pd.Series((torch.tensor(preds.predictions) > 0.98).squeeze().to(int)).map(
    id2label
)

# %%
df_test = pd.read_csv(TEST_PATH)
df_test["label"] = final_preds
df_test[["line_id", "label"]].to_csv("submission.csv", sep=",", index=False)
