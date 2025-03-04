import itertools
from collections import Counter, defaultdict
from math import log
from typing import Iterable, Optional, List, Dict


def tokenize_and_pmimask(text, span_length, pct, ceil_pct = False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text



def count_total_ngrams_per_size(text: str, max_ngram_size: int)-> Dict[int, int]:
    """_summary_

    Parameters
    ----------
    text : str
        Input texts
        
    max_ngram_size : int
        Max size of n

    Returns
    -------
    Dict[int, int]
        _description_
    """
    
    tokens = text.split(' ')
    total_number_of_ngrams_per_size = Counter()
    
    for ngram_size in range(1, max_ngram_size + 1):
        if len(tokens) >= ngram_size:
            total_number_of_ngrams_per_size[ngram_size] += len(tokens) - ngram_size + 1

    return dict(total_number_of_ngrams_per_size)


def count_ngrams(text: str, max_ngram_size: int) -> Dict[tuple, int]:
    """Counts individual ngrams in the input.

    :param tokenized_samples: the tokenized input sequences.
    :param max_ngram_size: the maximal ngram size to consider
    :return: dictionary mapping from ngram to the number of times it appears in the input.
    """
    tokens = text.split(' ')
    ngram_to_count = Counter()
    
    for ngram_size in range(1, max_ngram_size + 1):
        for start_i in range(len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[start_i:start_i+ngram_size])
            ngram_to_count[ngram] += 1
    return dict(ngram_to_count)

def compute_log_likelihood(ngram_to_count: Dict[tuple, int],
                           total_ngrams_per_size: Dict[int, int]) -> Dict[tuple, float]:
    """Computes the log likelihood of ngrams.

    :param ngram_to_count: dictionary mapping from ngrams to the number of times that they appear in the corpus.
    :param total_ngrams_per_size: dictionary mapping ngram sizes to the number of ngrams of that size.
    :return: a dictionary mapping ngrams to their log likelihood.
    """
    ngram_to_log_likelihood = {}
    for ngram, ngram_count in ngram_to_count.items():
        ngram_size = len(ngram)
        ngram_probability = ngram_count / total_ngrams_per_size[ngram_size]
        ngram_to_log_likelihood[ngram] = log(ngram_probability)

    return ngram_to_log_likelihood




if __name__ == '__main__':
    sample_text = "In the bustling city streets, amidst the cacophony of honking cars and chatter of passersby, stands an old, quaint bookstore. Its weathered facade, adorned with ivy creeping up the walls, exudes an air of timeless charm that draws in book enthusiasts from all walks of life. Stepping inside, you are enveloped by the rich aroma of aging paper and ink, and the soft creaking of the wooden floor beneath your feet. The shelves, lined with literary treasures from around the world, beckon you to embark on a literary journey like no other."
    
    a = count_total_ngrams_per_size(sample_text, 5)
    b = count_ngrams(sample_text, 5)
    c = compute_log_likelihood(b, a)
    
    print(a)
    print(b)
    print(c)

