import pandas as pd
import sys
from collections import Counter
from tqdm import tqdm
import json


if __name__ == '__main__':
    filepath = '../datasets/raw/in_domain_train.tsv'
    output_word_cab = '../datasets/raw/wordlist.txt'
    df = pd.read_csv(filepath, sep='\t', header=0)

    word_list_counter = Counter()

    bar1 = tqdm(desc='Get sentences', total=len(df))
    sentence_List = []
    label_List = []
    for i in range(len(df)):
        label1, verdict, human, sentences = df.iloc[i]
        label_List.append(human * 2 + verdict)
        sentence_List.append(sentences)
        bar1.update()

    bar1.close()
    reformed_list = []
    bar2 = tqdm(desc='Build Vocabs', total=len(df))
    for i in sentence_List:
        word_list = i.split(' ')
        for j in word_list:
            word_list_counter[j] = word_list_counter[j] + 1
        bar2.update()
    bar2.close()

    word_rev = {}
    word_list_cnt_list = word_list_counter.most_common()
    print('Vocab Size: ', len(word_list_cnt_list))
    for i, d in enumerate(word_list_cnt_list):
        word_rev[d[0]] = i

    bar3 = tqdm(desc='Transform', total=len(df))
    for i in range(len(df)):
        sentence_word = sentence_List[i].split(' ')
        new_sentence = json.dumps([word_rev[d] for d in sentence_word])
        reformed_list.append(new_sentence)
        bar3.update()

    df = pd.DataFrame({'data': reformed_list, 'label': label_List})
    df.to_csv('../datasets/raw/in_domain_train.reformed.csv')
    output_str = ['{}\t{}'.format(d[0], word_list_counter[d[0]]) for d in word_list_cnt_list]
    open(output_word_cab, 'w').write('\n'.join(output_str))
