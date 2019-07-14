#!/usr/bin/env python3

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords


def get_newline_index(s):
    j = 0
    for i, c in enumerate(s):
        if (c == '\n') or (c == '\r'):
            j = i
            break
    return ''.join([i for i in s[j:]
                    if i in string.printable and
                    (i not in string.punctuation)])


def main(fileName):
    df = pd.read_csv(fileName, encoding='iso-8859-1')
    # Column names are Speech_text, month, year
    df['Speech_text'] = [get_newline_index(t)
                         for t in df['Speech_text']]
    # Tokenize words
    speeches = df['Speech_text']  # Series of speeches
    speeches_time = []
    for speech in speeches:
        speech_tokens = nltk.word_tokenize(speech)
        # Now remove the stopwords
        speech_tokens = [w.lower() for w in speech_tokens
                         if w.lower() not in stopwords.words('english')]
        speech_tokens = nltk.pos_tag(speech_tokens)
        speech_tokens = [w for w, t in speech_tokens
                         if t == 'NN' or t == 'NNS']
        # Now make a Histogram of the most used words in each speech
        speech_dist = nltk.FreqDist(speech_tokens)
        # put words in order across all speeches
        # print(speech_dist.most_common(5))
        speeches_time.append(speech_dist.most_common(10))
    # Flattened
    speeches_time = [j for i in speeches_time
                     for j in i]
    # Now count the number of words across time series
    final_dict = {k: 0
                  for k in set([i for i, j in speeches_time])}
    for k in final_dict.keys():
        final_dict[k] = sum([c for w, c in speeches_time
                             if w == k])

    most_talked = (sorted(list(final_dict.items()), key=lambda v: v[1],
                          reverse=True))[:10]
    # Modi talks most about these 10 things over time
    print(most_talked)


if __name__ == '__main__':
    main('./mann-ki-baat-speech-corpus.zip')
