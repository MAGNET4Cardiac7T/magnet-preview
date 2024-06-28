import numpy as np

from typing import List, Dict




def gather_predictions(predictions: List[Dict[str, np.ndarray]]):
    collated_results = {}

    for key in predictions[0].keys():
        collated_results[key] = np.concatenate([batch[key] for batch in predictions])

    return collated_results