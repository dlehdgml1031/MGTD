from utils.generate_text import generate_text

if __name__ == '__main__':
    generate_model_name = 'gpt2-xl'
    batch_size = 16
    for dataset_name in ['xsum', 'imdb', 'squad', 'finance', 'wirting']:
        for prompt_token_length in [30, 60, 90]:
            for max_length in [250, 512]:
                generate_text(
                    generate_model_name = generate_model_name,
                    dataset = dataset_name,
                    n_samples = 5000,
                    batch_size = batch_size,
                    prompt_token_length = prompt_token_length,
                    max_length = max_length,
                    penalty_alpha = 0.6
                )