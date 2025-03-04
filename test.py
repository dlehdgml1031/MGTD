from MGTD import test_model

import argparse
import pandas as pd

if __name__ == '__main__':
    for dataset_name in ['imdb']:
        # for prompt_token_len in [30, 60, 90]:
        for prompt_token_len in [90]:
            for gen_max_len in [512]:
                # for train_ratio in [0.2, 0.5, 0.8]:
                for train_ratio in [0.8]:
                    base_mode_name = 'bert-base-uncased'
                    sample_size = 10000
                    train_size = int(sample_size * train_ratio)
                    test_size = sample_size - train_size
                    generate_model_name = "gpt2-xl"
                    batch_size = 16
                    if train_ratio == 0.2:
                        checkpoint_num = 500
                    elif train_ratio == 0.5:
                        checkpoint_num = 1250
                    elif train_ratio == 0.8:
                        checkpoint_num = 2000
                    at_model_dir = f'./results/at_0.1/{base_mode_name}_{dataset_name}_{generate_model_name}_inputTokenLen{prompt_token_len}_maxLen{gen_max_len}_trainsize{train_size}_testsize{test_size}/checkpoint-{checkpoint_num}'
                    # base_model_log = f'{base_mode_name}_{dataset_name}_{generate_model_name}_inputTokenLen{prompt_token_len}_maxLen{gen_max_len}_trainsize{train_size}_testsize{test_size}/checkpoint-{checkpoint_num}'
                    
                    test_metric = test_model(
                        test_model_name_or_path = at_model_dir,
                        generate_model_name = generate_model_name,
                        dataset_name = dataset_name,
                        sample_size = sample_size,
                        batch_size = batch_size,
                        n_train = train_size,
                        n_test = test_size,
                        prompt_token_len = prompt_token_len,
                        gen_max_len = gen_max_len,
                    )
                    
                    print('\n')
                    print(f'Test Model: {at_model_dir}')
                    # print(f'Test Model: {base_model_log}')
                    print('#' * 200)
                    print(test_metric)
                    print('#' * 200, end = '\n\n')
                    
                    
                    
                    
    
    
    
    
    
    
        
        
        
        
        
        
        
        

        
 