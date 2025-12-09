import numpy as np
from collections import Counter
import array
import ETCPy.ETC.CCMC.pairs as pairs

def symbolic_to_array(seq_str):
    unique_symbols = sorted(list(set(seq_str)))
    symbol_map = {symbol: i+1 for i, symbol in enumerate(unique_symbols)}

    int_list = [symbol_map[symbol] for symbol in seq_str]

    return array.array('I', int_list)

def get_ccm_results(Xs, Ys, penalty_threshold=1, efficacy_tolerance=0):
    x_array = symbolic_to_array(Xs)
    y_array = symbolic_to_array(Ys)

    results = pairs.CCM_causality(
        x_array,
        y_array,
        penalty_threshold=penalty_threshold,
        efficacy_tolerance=efficacy_tolerance
    )
    return results