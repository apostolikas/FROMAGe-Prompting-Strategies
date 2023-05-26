from eval_metrics.cider import Cider
from eval_metrics.rouge import Rouge
from eval_metrics.bleu import Bleu
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



if __name__ == '__main__':
        
    augmented = pd.read_csv('flickr_vis_augm1.txt', header=None, delimiter='\r\n')
    unaugmented = pd.read_csv('flickr_vis_augm2.txt', header=None, delimiter='\r\n')
    answers = pd.read_csv('flickr_vis_augm3.txt', header=None, delimiter='\r\n')

    unaugmented_refs = {}
    augmented_refs = {}
    targets = {}

    for i in range(len(augmented)):
        unaugmented_refs[i] = [(unaugmented.iloc[i,0])]
        augmented_refs[i] = [(augmented.iloc[i,0])]
        targets[i] = [(answers.iloc[i,0])]
        
    print('Printing scores for the case of not using any augmentation:')
    cider(unaugmented_refs,targets)
    rouge(unaugmented_refs,targets)
    bleu(unaugmented_refs,targets)
    print('\n')
    print('Printing scores for the case of using visual augmentation:')
    cider(augmented_refs,targets)
    rouge(augmented_refs,targets)
    bleu(augmented_refs,targets)


