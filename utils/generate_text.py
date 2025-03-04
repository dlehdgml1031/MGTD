import re
import random
import torch
import datasets
import numpy as np
import pandas as pd
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union


def _strip_newlines(text):
    return ' '.join(text.split())

def _process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')

def _process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()

        
def _trim_text(text):
    pattern = re.compile(r'\. ')
    if (text[-1] != '.') and (len(re.split(pattern, text)) > 2):
        text = '. '.join(re.split(pattern, text)[:-1])
    return text
    

def _load_writing_for_generation():
    data = datasets.load_dataset('euclaise/writingprompts', split = 'train')
    prompts = data['prompt']
    stories = data['story']
    
    prompts = [_process_prompt(prompt) for prompt in prompts]
    joined = [_process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def _load_finance_for_generation():
    finance_data1 = pd.read_csv('data/finance/financial_news.csv')['Text'].to_list()
    finance_data2 = pd.read_csv('data/finance/financial_news2.csv')['Text'].to_list()
    
    return finance_data1 + finance_data2


def _load_pubmed_for_generation():
    seperator = '<<<SEP>>>'
    data = datasets.load_dataset('pubmed_qa', 'pqa_unlabeled', split = 'train')
    data = [f'Question: {q} Answer:{seperator}{a}' for q, a in zip(data['question'], data['long_answer'])]
    return data
    

def _generate_samples(
    raw_data:List,
    batch_size:int,
    generate_model_name:str,
    generate_kwargs:dict,
    prompt_token_length:int
):
    
    # fix the seed at 42
    torch.manual_seed(42)
    np.random.seed(42)
    
    data = {
        "original": [],
        "sampled": [],
    }
    
    if 'gpt2' in generate_model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(generate_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(generate_model_name).to('cuda')
        
        for batch in range((len(raw_data) // batch_size) + 1):
            print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            
            try:
                encoded_input = tokenizer(original_text, return_tensors = "pt", padding = True).to('cuda')
                encoded_input = {key: value[:, :prompt_token_length] for key, value in encoded_input.items()}
            except:
                encoded_input = tokenizer(original_text, return_tensors = "pt", padding = True).to('cuda')
                encoded_input = {key: value[:, :30] for key, value in encoded_input.items()}
            
            # generate text
            with torch.no_grad():
                outputs = model.generate(
                    **encoded_input,
                    **generate_kwargs,
                    pad_token_id = tokenizer.eos_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)
            data['original'].extend(original_text)
            data['sampled'].extend(decoded)
    
    if 'llama' in generate_model_name:
        tokenizer = AutoTokenizer.from_pretrained(generate_model_name, use_auth_token = True)
        tokenizer.pad_token = tokenizer.eos_token
        pipeline = transformers.pipeline(
            'text-generation',
            model = generate_model_name,
            tokenizer = tokenizer,
            torch_dtype = torch.float16,
            device_map = 'auto'
        )
        
        for batch in range((len(raw_data) // batch_size) + 1):
            print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
    
            encoded_input = tokenizer(original_text, return_tensors = "pt", padding = True)
            encoded_input = [x[:prompt_token_length] for x in encoded_input['input_ids']]
            texts = tokenizer.batch_decode(encoded_input, skip_special_tokens = True)
            
            outputs = pipeline(
                texts,
                eos_token_id = tokenizer.eos_token_id,
                **generate_kwargs
            )
            
            outputs = [x[0]['generated_text'] for x in outputs]
            data['original'].extend(original_text)
            data['sampled'].extend(outputs)
    
    return data

def generate_text(
    generate_model_name:str,
    dataset:str,
    n_samples:int,
    batch_size:int,
    prompt_token_length:int,
    max_length:int,
    top_p:float = 0.96,
    top_k:int = 40,
    min_length:int = 150,
    do_sample:bool = True,
    repetition_penalty:float = 1.5,
    no_repeat_ngram_size:int = 3,
    penalty_alpha:Optional[float] = None,
):
    
    # load dataset
    if dataset == 'xsum':
        data = datasets.load_dataset('xsum', split = 'train')['document']
    elif dataset == 'squad':
        data = datasets.load_dataset('squad', split = 'train')['context']
    elif dataset == 'imdb':
        data = datasets.load_dataset('imdb', split = 'unsupervised')['text']
    elif dataset == 'writing':
        data = _load_writing_for_generation()
    elif dataset == 'finance':
        data = _load_finance_for_generation()
    elif dataset == 'pubmed':
        pass
    
    # remove duplicates from the data
    data = list(dict.fromkeys(data))
    
    # strip whitespace and remove newlines from each example
    data = [x.strip() for x in data]
    data = [_strip_newlines(x) for x in data]
    
    # try to keep only examples with > 250, 150, 100, 70
    if dataset in ['xsum', 'finance', 'writing']:
        long_data = [x for x in data if len(x.split()) > 250]
    elif dataset in ['imdb']:
        long_data = [x for x in data if len(x.split()) > 150]
    elif dataset in ['squad']:
        long_data = [x for x in data if len(x.split()) > 100]
    elif dataset in ['pubmed']:
        long_data = [x for x in data if len(x.split()) > 70]
    
    if len(long_data) > 0:
        data = long_data
    
    random.seed(0)
    random.shuffle(data)
    
    if dataset == 'xsum':
        data = data[:50_000]
    elif dataset == 'writing':
        data = data[:60_000]
    
    # keep exapmles with <= 512 tokens using the t5-small tokenizer
    preproc_tokenizer = AutoTokenizer.from_pretrained('t5-small', model_max_length = 512)
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    
    print('Number of examples in dataset:', len(data))
    assert len(data) >= (n_samples * 2), f'Dataset must have at least {n_samples * 2} examples but only has {len(data)}'
    
    if 'gpt2' in generate_model_name:
        if penalty_alpha:
            generate_kwargs = {
                'top_k' : top_k,
                'do_sample' : do_sample,
                'min_length' : min_length,
                'max_length' : max_length,
                'penalty_alpha' : penalty_alpha,
                'temperature' : 1.3,
            }
        else:
            generate_kwargs = {
                'top_p' : top_p,
                'top_k' : top_k,
                'do_sample' : do_sample,
                'min_length' : min_length,
                'max_length' : max_length,
                'repetition_penalty' : repetition_penalty,
                'no_repeat_ngram_size' : no_repeat_ngram_size,
                'temperature' : 1.3,
            }
    
    if 'llama' in generate_model_name:
        generate_kwargs = {
            'do_sample' : do_sample,
            'min_length' : min_length,
            'max_length' : max_length,
            'top_k' :  30,
            'top_p' : 0.96,
            'num_return_sequences' : 1,
        }
    
    print(f'Generate Model: {generate_model_name}\nDataset: {dataset}\nNum of Generation Texts: {n_samples}\nBatch Size: {batch_size}\nPrompt Token Lengths: {prompt_token_length}\nMax Len: {max_length}\nGenerate Kwargs: {generate_kwargs}')
    gen_text = _generate_samples(
        raw_data = data[:n_samples],
        batch_size = batch_size,
        generate_model_name = generate_model_name,
        prompt_token_length = prompt_token_length,
        generate_kwargs = generate_kwargs,
    )['sampled']
    
    assert len(gen_text) == n_samples, f'Generated {len(gen_text)} samples, but expected {n_samples}'
    
    human_df = pd.DataFrame({'text' : data[n_samples:n_samples * 2], 'label' : [0] * len(data[n_samples:n_samples * 2])})
    machine_df = pd.DataFrame({'text' : gen_text, 'label' : [1] * len(gen_text)})
    machine_df['text'] = machine_df['text'].apply(_trim_text)
    total_df = pd.concat([human_df, machine_df]).reset_index(drop = True)
    
    if 'llama' in generate_model_name:
        total_df.to_csv(f'data/{dataset}_llama2-7b_{n_samples * 2}.csv', index = False, encoding = 'utf-8')
    else:
        total_df.to_csv(f'data/{dataset}_{generate_model_name}_{n_samples * 2}_inputTokenLen_{prompt_token_length}_maxLen_{max_length}.csv', index = False, encoding = 'utf-8')
    
    return gen_text