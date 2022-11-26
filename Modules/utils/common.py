from typing import *

def print_progress_bar(prefix: str, rate: float, suffix: str, bar_length: int = 50) -> None:
    '''
    Print progress bar

    Parameter
    ----------
    - prefix: str, prefix of progress bar
    - rate: float, rate of progress bar
    - suffix: str, suffix of progress bar
    - bar_length: int, `default=50`, length of bar when printing

    Example
    ---------
    >>> print_progress_bar('epoch 9000', 0.5, '50%')
    # epoch 9000:	|████████████████████████----------------------|	50%
    '''
    bar_length = max(20, bar_length)
    length = int(rate * bar_length) + 1
    bar = '█' * length + '-' * max(0, bar_length - length)
    print(f'\r{prefix}:\t|{bar}|\t{suffix}', end='\r')