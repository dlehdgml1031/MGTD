import json
import random
import datasets
import pandas as pd

def _preporcess_text(text):
    text = text.replace(
        '\n', '').replace(
        '\t', '').replace(
        '\r', '').replace(
        "\'", "").replace(
        '-', '').replace(
        '.', '').replace(
        ',', '').replace(
        '?', '').replace(
        '!', '').replace(
        '  ', ' ').replace(
        '   ', ' ').strip()
    
    return ' '.join(text.split())

def load_data(
    file_path:str,
    n_train:int,
    n_test:int,
):
    df = pd.read_csv(file_path)
    data_len = len(df['text'])
    df['text'] = df['text'].apply(_preporcess_text)
    data = datasets.Dataset.from_dict({'text' : df['text'].to_list(), 'label' : df['label'].to_list()})
    data = data.class_encode_column("label")
    
    # split train test with seed 0
    return datasets.DatasetDict(data.train_test_split(train_size = n_train/data_len, test_size = n_test/data_len, shuffle = True, stratify_by_column = "label", seed = 0))

def load_hc3(
    file_path:str
):
    data_list = []
    with open(file_path, 'r', encoding = 'utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data_list.append(json_object)
        
    df = pd.DataFrame(data_list)
    df['human_answers'] = df['human_answers'].apply(lambda x: _preporcess_text(' '.join(x)))
    df['chatgpt_answers'] = df['chatgpt_answers'].apply(lambda x: _preporcess_text(' '.join(x)))
    
    df_human = {
        'text' : df['human_answers'].to_list(),
        'label' : [0] * len(df['human_answers'].to_list())
    }
    
    df_machine = {
        'text' : df['chatgpt_answers'].to_list(),
        'label' : [1] * len(df['chatgpt_answers'].to_list())
    }
    
    df_human = pd.DataFrame(df_human)
    df_machine = pd.DataFrame(df_machine)
    
    df = pd.concat([df_human, df_machine])
    df = df.reset_index(drop = True)
    
    data = datasets.Dataset.from_dict({'text' : df['text'].to_list(), 'label' : df['label'].to_list()})
    data = data.class_encode_column("label")
    
    return datasets.DatasetDict(data.train_test_split(train_size = 0.8, test_size = 0.2, shuffle = True, stratify_by_column = "label", seed = 0))
