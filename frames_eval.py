import json

import torch
import numpy as np
import pandas as pd
from torchmetrics.text.bert import BERTScore

device = "cuda" if torch.cuda.is_available() else "cpu"

results = pd.read_csv('frames/results.csv', sep=';', encoding='utf8')

preds = results['model_caption'].tolist()
target = results['gif_caption'].tolist()

bertscore = BERTScore(device=device)
scores = bertscore(preds, target)

with open('frames/scores.json', 'w') as f:
    json.dump(scores, f)

avg_scores = {}
avg_scores['precision'] = np.mean(scores['precision']).item()
avg_scores['recall'] = np.mean(scores['recall']).item()
avg_scores['f1'] = np.mean(scores['f1']).item()

with open('frames/avg_scores.json', 'w') as f:
    json.dump(avg_scores, f)
