from MGTD import FineTuningBaseModel, AdversarialTraining, FineTuningHC3, AdversarialTrainingHC3

from typing import List
import time
import torch

def train_model(
    base_model_name_list: List,
    dataset_name_list: List,
    generate_model_name:str,
    mask_filling_model_name:str,
    gen_max_len:int,
    prompt_token_len_list:List,
    train_ratio_list:List,
    batch_size:int = 8,
    epoch:int = 2,
    learning_rate:float = 1e-4,
):
    
    # train base model
    for base_model_name in base_model_name_list:
        for dataset_name in dataset_name_list:
            for prompt_token_len in prompt_token_len_list:
                for train_ratio in train_ratio_list:
                    project_name = f'MGTD-{dataset_name}-ver2'
                    sample_size = 10000
                    train_size = int(10000 * train_ratio)
                    test_size = 10000 - train_size
                    
                    torch.cuda.empty_cache()
                    mgtd = FineTuningBaseModel(
                        base_model_name = base_model_name,
                        dataset_name = dataset_name,
                        generate_model_name = generate_model_name
                    )
                    
                    mgtd.train_base_model(
                        sample_size = sample_size,
                        n_train = train_size,
                        n_test = test_size,
                        batch_size = batch_size,
                        learning_rate = learning_rate,
                        epoch = epoch,
                        project_name = project_name,
                        prompt_token_len = prompt_token_len,
                        gen_max_len = gen_max_len
                    )
                    
                    time.sleep(5)
                    del mgtd
                    time.sleep(5)
    
    # train at model
    for base_model_name in base_model_name_list:
        for dataset_name in dataset_name_list:
            for prompt_token_len in prompt_token_len_list:
                for train_ratio in train_ratio_list:
                    project_name = f'MGTD-{dataset_name}-ver2'
                    sample_size = 10000
                    train_size = int(10000 * train_ratio)
                    test_size = 10000 - train_size
                    
                    torch.cuda.empty_cache()
                    mgtd = AdversarialTraining(
                        base_model_name = base_model_name,
                        mask_filling_model_name = mask_filling_model_name,
                        generate_model_name = generate_model_name,
                        dataset_name = dataset_name,
                    )
                    
                    mgtd.adversarial_train_base_model(
                        sample_size = sample_size,
                        n_train = train_size,
                        n_test = test_size,
                        batch_size = batch_size,
                        epoch = epoch,
                        learning_rate = learning_rate,
                        project_name = project_name,
                        prompt_token_len = prompt_token_len,
                        gen_max_len = gen_max_len,
                        weight_alpha = 0.1
                    )
                    
                    time.sleep(5)
                    del mgtd
                    time.sleep(5)


def train_hc3_model(
    base_model_name_list: List,
    dataset_name_list: List,
    mask_filling_model_name:str,
    batch_size:int = 8,
    epoch:int = 2,
    learning_rate:float = 1e-4,
):
    
    # train base model
    for base_model_name in base_model_name_list:
        for dataset_name in dataset_name_list:
            project_name = f'MGTD-HC3'
            
            torch.cuda.empty_cache()
            mgtd = FineTuningHC3(
                base_model_name = base_model_name,
                dataset_name = dataset_name,
            )
            
            mgtd.train_base_model(
                batch_size = batch_size,
                learning_rate = learning_rate,
                epoch = epoch,
                project_name = project_name,
            )
            
            time.sleep(5)
            del mgtd
            time.sleep(5)
    
    # train at model
    for base_model_name in base_model_name_list:
        for dataset_name in dataset_name_list:
            project_name = f'MGTD-HC3'
            
            torch.cuda.empty_cache()
            mgtd = AdversarialTrainingHC3(
                base_model_name = base_model_name,
                mask_filling_model_name = mask_filling_model_name,
                dataset_name = dataset_name,
            )
            
            mgtd.adversarial_train_base_model(
                batch_size = batch_size,
                epoch = epoch,
                learning_rate = learning_rate,
                project_name = project_name,
                weight_alpha = 0.1
            )
            
            time.sleep(5)
            del mgtd
            time.sleep(5)



if __name__ == '__main__':
    # train_model(
    #     base_model_name_list = ['bert-base-uncased', 'roberta-base'],
    #     dataset_name_list = ['xsum'],
    #     generate_model_name = "gpt2-xl",
    #     mask_filling_model_name = "t5-small",
    #     gen_max_len = 512,
    #     prompt_token_len_list = [30, 60, 90],
    #     train_ratio_list = [0.2, 0.5, 0.8],
    #     batch_size = 8,
    #     epoch = 2,
    #     learning_rate = 1e-4
    # )
    
    train_hc3_model(
        base_model_name_list = ['bert-base-uncased'],
        dataset_name_list = ['finance', 'medicine', 'open_qa', 'wiki_csai', 'reddit_eli5'],
        mask_filling_model_name = "t5-small",
        batch_size = 8,
        epoch = 2,
        learning_rate = 1e-4
    )