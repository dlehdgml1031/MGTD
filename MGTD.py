from utils import custom_datasets

import os
import re
import tqdm
import time
import itertools
from math import log
from collections import Counter, defaultdict
from typing import Iterable, Optional, List, Dict

import torch
import wandb
import evaluate
import numpy as np
import transformers
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class FineTuningBaseModel(object):
    def __init__(self,
                 base_model_name:str,
                 dataset_name:str,
                 generate_model_name:str,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model_name = base_model_name
        self.dataset_name = dataset_name
        self.generate_model_name = generate_model_name
    
    def train_base_model(
        self,
        sample_size:int,
        n_train:int,
        n_test:int,
        batch_size:int,
        learning_rate:float,
        epoch:int,
        project_name:str,
        prompt_token_len:int,
        gen_max_len:int,
    ):
        # login wandb
        wandb.login()
        os.environ['WANDB_PROJECT'] = project_name
        
        # load base model
        base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, max_length = 512, padding = True, truncation = True)
        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, num_labels = 2).to(self.device)
        
        # load dataset
        if self.dataset_name == 'total':
            data = custom_datasets.load_data(
                f'./data/{self.dataset_name}_{self.generate_model_name}_{sample_size}_inputTokenLen_{prompt_token_len}_maxLen_{gen_max_len}.csv',
                n_train = n_train,
                n_test = n_test
            )
        
        else:
            data = custom_datasets.load_data(
                f'./data/{self.dataset_name}_{self.generate_model_name}_{sample_size}_inputTokenLen_{prompt_token_len}_maxLen_{gen_max_len}.csv',
                n_train = n_train,
                n_test = n_test
            )
        
        # tokenized text
        tokenized_data = data.map(lambda examples: base_tokenizer(examples['text'], truncation = True), batched = True)
        data_collator = transformers.DataCollatorWithPadding(tokenizer = base_tokenizer)
        
        # netsted function for compute metrics
        def _compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'binary')
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, preds)
            return {
                "accuracy" : acc,
                "f1" : f1,
                "precision" : precision,
                "recall" : recall,
                'auroc' : auc
            }
        
        # define training arguments
        # using AdamW optimizer
        training_args = transformers.TrainingArguments(
            output_dir = f'./results/ft/{self.base_model_name}_{self.dataset_name}_{self.generate_model_name}_inputTokenLen{prompt_token_len}_maxLen{gen_max_len}_trainsize{n_train}_testsize{n_test}',
            run_name = f'{self.base_model_name}-{self.dataset_name}-{self.generate_model_name}-inputTokenLen{prompt_token_len}-maxLen{gen_max_len}-trainsize{n_train}-testsize{n_test}',
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            learning_rate = learning_rate,
            weight_decay = 0.01,
            num_train_epochs = epoch,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end = True,
            report_to = 'wandb',
            logging_dir = './logs',
            logging_steps = 10,
            seed = 0
        )
        
        trainer = transformers.Trainer(
            model = base_model,
            tokenizer = base_tokenizer,
            args = training_args,
            train_dataset = tokenized_data['train'],
            eval_dataset = tokenized_data['test'],
            data_collator = data_collator,
            compute_metrics = _compute_metrics,
        )
        
        trainer.train()
        
        wandb.finish()


class FineTuningHC3(object):
    def __init__(self,
                 base_model_name:str,
                 dataset_name:str,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model_name = base_model_name
        self.dataset_name = dataset_name
    
    def train_base_model(
        self,
        batch_size:int,
        learning_rate:float,
        epoch:int,
        project_name:str,
    ):
        # login wandb
        wandb.login()
        os.environ['WANDB_PROJECT'] = project_name
        
        # load base model
        base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, max_length = 512, padding = True, truncation = True)
        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, num_labels = 2).to(self.device)
        
        data = custom_datasets.load_hc3(f'./data/HC3/{self.dataset_name}.jsonl')
        
        # tokenized text
        tokenized_data = data.map(lambda examples: base_tokenizer(examples['text'], truncation = True), batched = True)
        data_collator = transformers.DataCollatorWithPadding(tokenizer = base_tokenizer)
        
        # netsted function for compute metrics
        def _compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'binary')
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, preds)
            return {
                "accuracy" : acc,
                "f1" : f1,
                "precision" : precision,
                "recall" : recall,
                'auroc' : auc
            }
        
        # define training arguments
        # using AdamW optimizer
        training_args = transformers.TrainingArguments(
            output_dir = f'./results/ft/{self.base_model_name}_HC3_{self.dataset_name}',
            run_name = f'{self.base_model_name}-HC3_{self.dataset_name}',
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            learning_rate = learning_rate,
            weight_decay = 0.01,
            num_train_epochs = epoch,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end = True,
            report_to = 'wandb',
            logging_dir = './logs',
            logging_steps = 10,
            seed = 0
        )
        
        trainer = transformers.Trainer(
            model = base_model,
            tokenizer = base_tokenizer,
            args = training_args,
            train_dataset = tokenized_data['train'],
            eval_dataset = tokenized_data['test'],
            data_collator = data_collator,
            compute_metrics = _compute_metrics,
        )
        
        trainer.train()
        
        wandb.finish()


class PerturbationModel(object):
    def __init__(
        self,
        mask_filling_model_name:str,
        mask_top_p_rate:float = 0.96,
        pct_size:float = 0.3,
        span_length:int = 2,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mask_filling_model_name:str = mask_filling_model_name
        self.mask_top_p_rate:float = mask_top_p_rate
        self.pct_size:float = pct_size
        self.span_length:int = span_length
    
    def _tokenize_and_mask(
        self,
        text:str,
        span_length:int,
        pct:float,
        buffer_size:int,
        celi_pct:Optional[bool] = False
    ):
        tokens = text.split()
        mask_string = '<<<mask>>>'
        n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
        
        if celi_pct:
            n_spans = np.ceil(n_spans)
        
        n_spans = int(n_spans)
        
        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - buffer_size)
            search_end = min(len(tokens), end + buffer_size)
            
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each mask_string with <extra_id_NUM>
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        
        return text
    
    def _count_masks(self, texts:List):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    
    # replace each masked span with a sample from T5 mask_model
    def _replace_masks(
        self,
        texts,
        mask_model,
        mask_tokenizer,
        mask_top_p_rate:float
    ):
        n_expected = self._count_masks(texts)
        stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = mask_tokenizer(texts, return_tensors = "pt", padding = True).to(self.device)
        with torch.no_grad():
            outputs = mask_model.generate(**tokens, max_length = 512, do_sample = True, top_p = mask_top_p_rate, num_return_sequences = 1, eos_token_id = stop_id)
        return mask_tokenizer.batch_decode(outputs, skip_special_tokens = False)
    
    def _extract_fills(self, texts: List):
        # define pattern
        pattern = re.compile(r"<extra_id_\d+>")
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        # return the text in between each matched mask token
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]    
        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]    
        return extracted_fills
    
    
    def _apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]
        n_expected = self._count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                try:
                    for fill_idx in range(len(fills)):
                        text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
                except:
                    pass
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
        
        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts


    def perturb_text(
        self,
        texts:List,
        ceil_pct:Optional[bool] = False
    ):
        mask_model = T5ForConditionalGeneration.from_pretrained(self.mask_filling_model_name).to(self.device)
        mask_tokenizer = T5Tokenizer.from_pretrained(self.mask_filling_model_name, model_max_length = 512)
        
        masked_texts = [self._tokenize_and_mask(text = x, span_length =  self.span_length, pct = self.pct_size, buffer_size = 1, celi_pct =  ceil_pct) for x in texts]
        raw_fills = self._replace_masks(texts = masked_texts, mask_model = mask_model, mask_tokenizer = mask_tokenizer, mask_top_p_rate = self.mask_top_p_rate)
        extracted_fills = self._extract_fills(raw_fills)
        perturbed_texts = self._apply_extracted_fills(masked_texts, extracted_fills)
        
        return perturbed_texts


class AdversarialTrainer(Trainer):
    def __init__(
        self,
        weight_alpha:float,
        PerturbationModel: PerturbationModel,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_alpha = weight_alpha
        self.PerturbationModel = PerturbationModel
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
    ):
        labels = inputs.pop("labels")
        original_texts = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens = True)
        adv_text = self.PerturbationModel.perturb_text(texts = original_texts)
        adv_inputs = self.tokenizer(adv_text, return_tensors = 'pt', padding=True, truncation=True).to('cuda')

        normal_outputs = model(**inputs)
        adv_outputs = model(**adv_inputs)
        ce_loss_func = torch.nn.CrossEntropyLoss()
        
        normal_loss = ce_loss_func(normal_outputs[0], labels)
        adv_loss = ce_loss_func(adv_outputs[0], labels)
        
        # total_loss = (self.weight_alpha * normal_loss) + ((1 - self.weight_alpha) * adv_loss)
        total_loss = normal_loss + (self.weight_alpha * adv_loss)
        
        return (total_loss, normal_outputs) if return_outputs else total_loss


class AdversarialTraining(object):
    def __init__(
        self,
        base_model_name:str,
        mask_filling_model_name:str,
        generate_model_name:str,
        dataset_name:str,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model_name = base_model_name
        self.mask_filling_model_name = mask_filling_model_name
        self.generate_model_name = generate_model_name
        self.dataset_name = dataset_name
    
    def adversarial_train_base_model(
        self,
        sample_size:int,
        n_train:int,
        n_test:int,
        batch_size:int,
        epoch:int,
        learning_rate:float,
        project_name:str,
        prompt_token_len:int,
        gen_max_len:int,
        weight_alpha:float,
    ):
        # login wandb
        wandb.login()
        os.environ['WANDB_PROJECT'] = project_name
        
        # load base model and tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, max_length = 512, padding = True, truncation = True)
        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, num_labels = 2).to(self.device)
        
        # load dataset
        if self.dataset_name == 'total':
            data = custom_datasets.load_data(
                f'./data/{self.dataset_name}_{self.generate_model_name}_{sample_size}_inputTokenLen_{prompt_token_len}_maxLen_{gen_max_len}.csv',
                n_train = n_train,
                n_test = n_test
            )
        
        else:
            data = custom_datasets.load_data(
                f'./data/{self.dataset_name}_{self.generate_model_name}_{sample_size}_inputTokenLen_{prompt_token_len}_maxLen_{gen_max_len}.csv',
                n_train = n_train,
                n_test = n_test
            )
        
        # tokenized text
        tokenized_data = data.map(lambda examples: base_tokenizer(examples['text'], truncation = True), batched = True)
        data_collator = transformers.DataCollatorWithPadding(tokenizer = base_tokenizer)
        
        # define training arguments
        # using AdamW optimizer
        training_args = transformers.TrainingArguments(
            output_dir = f'./results/at_{weight_alpha}/{self.base_model_name}_{self.dataset_name}_{self.generate_model_name}_inputTokenLen{prompt_token_len}_maxLen{gen_max_len}_trainsize{n_train}_testsize{n_test}',
            run_name = f'AT-weight{weight_alpha}-{self.base_model_name}-{self.dataset_name}-{self.generate_model_name}-inputTokenLen{prompt_token_len}-maxLen{gen_max_len}-trainsize{n_train}-testsize{n_test}',
            per_device_train_batch_size = batch_size,
            learning_rate = learning_rate,
            weight_decay = 0.01,
            num_train_epochs = epoch,
            save_strategy = "epoch",
            report_to = 'wandb',
            logging_dir = './logs',
            logging_steps = 10,
            seed = 0
        )
        
        perturbation_model = PerturbationModel(mask_filling_model_name = self.mask_filling_model_name)
            
        trainer = AdversarialTrainer(
            PerturbationModel = perturbation_model,
            weight_alpha = weight_alpha,
            model = base_model,
            tokenizer = base_tokenizer,
            args = training_args,
            train_dataset = tokenized_data['train'],
            data_collator = data_collator,
        )
        
        trainer.train()
        
        wandb.finish()


class AdversarialTrainingHC3(object):
    def __init__(
        self,
        base_model_name:str,
        mask_filling_model_name:str,
        dataset_name:str,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model_name = base_model_name
        self.mask_filling_model_name = mask_filling_model_name
        self.dataset_name = dataset_name
    
    def adversarial_train_base_model(
        self,
        batch_size:int,
        epoch:int,
        learning_rate:float,
        project_name:str,
        weight_alpha:float,
    ):
        # login wandb
        wandb.login()
        os.environ['WANDB_PROJECT'] = project_name
        
        # load base model and tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, max_length = 512, padding = True, truncation = True)
        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, num_labels = 2).to(self.device)
        
        data = custom_datasets.load_hc3(f'./data/HC3/{self.dataset_name}.jsonl')
        
        # tokenized text
        tokenized_data = data.map(lambda examples: base_tokenizer(examples['text'], truncation = True), batched = True)
        data_collator = transformers.DataCollatorWithPadding(tokenizer = base_tokenizer)
        
        # define training arguments
        # using AdamW optimizer
        training_args = transformers.TrainingArguments(
            output_dir = f'./results/at_{weight_alpha}/{self.base_model_name}_HC3_{self.dataset_name}',
            run_name = f'AT-weight{weight_alpha}-{self.base_model_name}-HC3_{self.dataset_name}',
            per_device_train_batch_size = batch_size,
            learning_rate = learning_rate,
            weight_decay = 0.01,
            num_train_epochs = epoch,
            save_strategy = "epoch",
            report_to = 'wandb',
            logging_dir = './logs',
            logging_steps = 10,
            seed = 0
        )
        
        perturbation_model = PerturbationModel(mask_filling_model_name = self.mask_filling_model_name)
            
        trainer = AdversarialTrainer(
            PerturbationModel = perturbation_model,
            weight_alpha = weight_alpha,
            model = base_model,
            tokenizer = base_tokenizer,
            args = training_args,
            train_dataset = tokenized_data['train'],
            data_collator = data_collator,
        )
        
        trainer.train()
        
        wandb.finish()
    

def test_model(
    test_model_name_or_path:str,
    generate_model_name:str,
    dataset_name:str,
    sample_size: int,
    batch_size: int,
    n_train: int,
    n_test: int,
    prompt_token_len:int,
    gen_max_len:int,
):
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = custom_datasets.load_data(
        f'./data/{dataset_name}_{generate_model_name}_{sample_size}_inputTokenLen_{prompt_token_len}_maxLen_{gen_max_len}.csv',
        n_train = n_train,
        n_test = n_test,
    )
    
    test_dataset = test_data['test']
    
    # load base model and tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(test_model_name_or_path, max_length = 512, padding = True, truncation = True)
    base_model = AutoModelForSequenceClassification.from_pretrained(test_model_name_or_path, num_labels = 2).to(device)
    
    def tokenize_function(examples):
        return base_tokenizer(examples["text"], padding = "max_length", truncation=True)
    
    test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
    test_tokenized_datasets = test_tokenized_datasets.remove_columns(["text"])
    test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")
    test_tokenized_datasets.set_format("torch")
    
    test_dataloader = DataLoader(test_tokenized_datasets, batch_size = batch_size)
    
    metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    base_model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = base_model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim = -1)
        metric.add_batch(predictions = predictions, references = batch["labels"])

    return metric.compute()
    
    
            
            
            
            
            
            