import pandas as pd
import sys
from collections import Counter
from tqdm import tqdm
import json


if __name__ == '__main__':
    filepath = '../datasets/tokenized/in_domain_dev.tsv'
    output_word_cab = '../datasets/tokenized/wordlist.txt'
    df = pd.read_csv(filepath, sep='\t', header=0)

    word_list_cnt = open(output_word_cab, 'r').readlines()
    word_list_dict = {d.split('\t')[0]: i for i, d in enumerate(word_list_cnt)}

    bar1 = tqdm(desc='Transform sentences', total=len(df))
    sentence_List = []
    label_List = []
    for i in range(len(df)):
        label1, verdict, human, sentences = df.iloc[i]
        label_List.append(human * 2 + verdict)
        word_sentence_list = sentences.split(' ')
        word_ap = []
        for word in word_sentence_list:
            if word in word_list_dict.keys():
                word_ap.append(word_list_dict[word])
            else:
                word_ap.append(len(word_list_dict))
        sentence_List.append(json.dumps(word_ap))
        bar1.update()

    df = pd.DataFrame({'data': sentence_List, 'label': label_List})
    df.to_csv('../datasets/tokenized/in_domain_dev.reformed.csv')
