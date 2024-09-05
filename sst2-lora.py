from datasets import load_dataset
from transformers import RobertaForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from copy import deepcopy
from torch.utils.data import random_split
import torch
import evaluate
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import Data
trainset = load_dataset("stanfordnlp/sst2", split = "train")
valset = load_dataset("stanfordnlp/sst2", split = "validation")

trainset.set_format("torch", device=device)
valset.set_format("torch", device=device)

split = random_split(valset, [len(valset)//2, len(valset)-len(valset)//2],
             generator = torch.Generator().manual_seed(42))
valset, testset = split[0], split[1]

# Set up tokenizer and model
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base",
                                                         num_labels=2, id2label=id2label, label2id=label2id)
## Apply LoRA
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, 
                         inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model = model.to(device)

def encode(examples):
    return tokenizer(examples["sentence"], truncation=True)

trainset = trainset.map(encode)
valset = valset.map(encode)

split = random_split(valset, [len(valset)//2, len(valset)-len(valset)//2],
             generator = torch.Generator().manual_seed(42))
valset, testset = split[0], split[1]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Define evaluation metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Trainer 
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    #per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Lists to store accuracies
train_accuracies = []
eval_accuracies = []
epoch = []

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
        
class CustomTrainer(Trainer):
    def log(self, logs):
        super().log(logs)
        if "eval_accuracy" in logs:
            eval_accuracies.append(logs["eval_accuracy"])
            epoch.append(logs["epoch"])
        if "train_accuracy" in logs:
            train_accuracies.append(logs["train_accuracy"])

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=trainset,
    eval_dataset=valset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.add_callback(CustomCallback(trainer)) 

trainer.train()

# trainer.evaluate(testset)
# trainer.evaluate(valset)