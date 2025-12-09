import numpy as np
from collections import Counter
import array
import ETCPy.ETC.CCMC.pairs as pairs

def symbolic_to_array(seq_str):
    unique_symbols = sorted(list(set(seq_str)))
    symbol_map = {symbol: i+1 for i, symbol in enumerate(unique_symbols)}

    int_list = [symbol_map[symbol] for symbol in seq_str]

    return array.array('I', int_list)

def causal_direction(Xs, Ys, analysis_print=False, penalty_threshold=1, efficacy_tolerance=0):
    x_array = symbolic_to_array(Xs)
    y_array = symbolic_to_array(Ys)

    results = pairs.CCM_causality(
        x_array,
        y_array,
        penalty_threshold=penalty_threshold,
        efficacy_tolerance=efficacy_tolerance
    )

    cause_map = {
        'x':1, # X causes Y
        'y':2, # Y causes X
        'n_or_m':3 # Undetermined
    }

    if results['Consensus']:
        cause = results['ETCP_cause']
        if cause in cause_map:
            return cause_map[cause]
        
        return 3
        
    causes = [results['ETCP_cause'], results['ETCE_cause'], results['LZP_cause']]
    # print(causes)
    counts = Counter(causes)
    # print(counts)
    
    if counts['x'] >= 2:
        return 1

    elif counts['y'] >= 2:
        return 2
    
    elif counts['n_or_m'] >= 2:
        return 3
    
    return 0