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

    refs = {}
    targets = {}

    for i in range(len(augmented)):
        refs[i] = [(unaugmented.iloc[i,0]),(augmented.iloc[i,0])]
        targets[i] = [(answers.iloc[i,0])]
        
    print('\n')
    cider(refs,targets)
    rouge(refs,targets)
    bleu(refs,targets)