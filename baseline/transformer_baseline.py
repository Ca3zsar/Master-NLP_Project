from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed, AutoConfig
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torch.optim import AdamW
from transformers import get_scheduler
from model_custom import CustomModel
torch.cuda.empty_cache()

# set max_split_size_mb 
os.environ['TOKENIZERS_PARALLELISM'] = "false"

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], max_length=512, truncation=True, padding='max_length')


def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df, val_df = train_test_split(train_df, test_size=0.9, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # check if checkpoints_path exists
    if not os.path.exists(checkpoints_path):
        # get tokenizer and model from huggingface
        tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
        tokenizer.model_max_len=512
        config = AutoConfig.from_pretrained(model)
        model = CustomModel(model, num_labels=len(label2id))
        
    else:
        # get tokenizer and model from saved model
        tokenizer = AutoTokenizer.from_pretrained(checkpoints_path)
        model = CustomModel(checkpoints_path, num_labels=len(label2id))
        config = AutoConfig.from_pretrained(checkpoints_path)
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    
    tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_valid_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_valid_dataset, batch_size=8, collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar_train.update(1)

        model.eval()
        for batch in eval_dataloader:
            batch = {k: v for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            f1_metric.add_batch(predictions=predictions, references=batch["labels"])
            acc_metric.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_eval.update(1)
        
        print(f"Epoch {epoch}")
        print(f1_metric.compute())
        print(acc_metric.compute())
    
    # save model
    model.save_pretrained(checkpoints_path)
    tokenizer.save_pretrained(checkpoints_path)

    # save config
    config.save_pretrained(checkpoints_path)


def test(test_df, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)

    args = parser.parse_args()

    random_seed = 0
    train_path =  args.train_file_path # For example 'subtaskA_train_multilingual.jsonl'
    test_path =  args.test_file_path # For example 'subtaskA_test_multilingual.jsonl'
    model =  args.model # For example 'xlm-roberta-base'
    subtask =  args.subtask # For example 'A'
    prediction_path = args.prediction_file_path # For example subtaskB_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}

    set_seed(random_seed)

    #get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)
    
    # train detector model
    fine_tune(train_df, valid_df, f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model)

    # test detector model
    results, predictions = test(test_df, f"model/subtask{subtask}/{random_seed}/best/", id2label, label2id)
    
    logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')
