import numpy as np
import json

def save_logits(logits, path):
    np.savez_compressed(path, logits,)

def save_report(report, path):
    with open(path, 'w') as fp:
         json.dump(report, fp, indent=6)