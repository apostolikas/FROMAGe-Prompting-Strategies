import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from src.video_captioning.scripts.eval_metrics.cider import Cider
from src.video_captioning.scripts.eval_metrics.rouge import Rouge
from src.video_captioning.scripts.eval_metrics.bleu import Bleu
import pandas as pd


def bleu(refs,labels):
    scorer = Bleu(n=4)
    score, scores = scorer.compute_score(refs,labels)
    print('bleu = %s' % score)


def cider(refs,labels):
    scorer = Cider()
    (score, scores) = scorer.compute_score(refs,labels)
    print('cider = %s' % score)


def rouge(refs,labels):
    scorer = Rouge()
    score, scores = scorer.compute_score(refs,labels)
    print('rouge = %s' % score)

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


results_path = 'src/video_captioning/experiments/uniform_5/results.csv'

results = pd.read_csv(results_path, sep=';', encoding='utf8')

preds = results['model_caption'].tolist()
targets = results['gif_caption'].tolist()

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
lm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

cos_sim_scores = []

for pred, target in zip(preds, targets):
     
    encoded_pred = tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
    encoded_target = tokenizer(target, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_pred = lm(**encoded_pred)
        model_target = lm(**encoded_target)

    pred_embeddings = F.normalize(mean_pooling(model_pred, encoded_pred['attention_mask']), p=2, dim=1)
    target_embeddings = F.normalize(mean_pooling(model_target, encoded_target['attention_mask']), p=2, dim=1)

    score = cos_sim(pred_embeddings, target_embeddings)
    cos_sim_scores.append(score.item())


print(results_path)
print('==================')
print(np.mean(cos_sim_scores))
print(np.std(cos_sim_scores))


refs = {}
targets_refs = {}

for i in range(len(targets)):
    refs[i] = [preds[i]]
    targets_refs[i] = [targets[i]]
    
print('\n')
cider(refs,targets_refs)
rouge(refs,targets_refs)
bleu(refs,targets_refs)
