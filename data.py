import os
import pandas as pd
import numpy as np
import emoji
import wordsegment
from config import OLID_PATH, GERMEVAL_PATH 
from utils import pad_sents, get_mask, get_lens
import re

DATASET_PATH = {
    'en': OLID_PATH,
    'de': GERMEVAL_PATH, 
}
DATASET_DICT = {
    'en': 'olid-training-v1.0.tsv',
    'de': 'germeval2018.training.txt',
    'de_ts' : 'germeval2018.test.txt',
    'en_ts' : 'germeval2018.test.txt',
}

wordsegment.load()

def read_file(filepath: str, data='en'):
    if data=='tain_en_test_en':
        df = pd.read_csv(filepath, sep='\t', keep_default_na=False)

        ids = np.array(df['id'].values)
        tweets = np.array(df['tweet'].values)

        # Process tweets
        tweets = process_tweets(tweets)
        label_a = np.array(df['subtask_a'].values)
        label_b = df['subtask_b'].values
        label_c = np.array(df['subtask_c'].values)

        nums = len(df)

    elif data=='train_de_test_de':
        df = pd.read_csv(filepath, sep='\t', keep_default_na=False, header=None)

        ids = np.array(range(1,len(df)+1))

        tweets = np.array(df[0].values)
        tweets = process_tweets(tweets)

        label_a = np.array(df[1].values)
        label_b = df[2].values
        label_c = None

        nums = len(df)

    elif data=='train_ende_test_ende':
        ## ENG
        filepath_en = os.path.join(DATASET_PATH['en'], DATASET_DICT['en']) 
        df_en = pd.read_csv(filepath_en, sep='\t', keep_default_na=False)

        ## DE
        filepath_de = os.path.join(DATASET_PATH['de'], DATASET_DICT['de'])
        df_de = pd.read_csv(filepath_de, sep='\t', keep_default_na=False, header=None)
        
        ## IDS
        ids_en = list(df_en['id'].values)
        ids_de = list(range(1,len(df_de)+1))
        ids = np.array(ids_en + ids_de)
        
        ## TWEETS
        tweets_en = list(df_en['tweet'].values)
        tweets_de = list(df_de[0].values)
        tweets = np.array(tweets_en + tweets_de)
        tweets = process_tweets(tweets)

        # Process tweets
        label_a_en = list(df_en['subtask_a'].values)
        label_a_de = list(df_de[1].values)
        label_a = np.array(label_a_en + label_a_de)

        nums = len(df_en) + len(df_de)


        label_b = None
        label_c = None

    elif data=='train_ende_test_en':
          ## ENG
        filepath_en = os.path.join(DATASET_PATH['en'], DATASET_DICT['en']) 
        df_en = pd.read_csv(filepath_en, sep='\t', keep_default_na=False)

        ## DE
        filepath_de = os.path.join(DATASET_PATH['de'], DATASET_DICT['de'])
        df_de = pd.read_csv(filepath_de, sep='\t', keep_default_na=False, header=None)
        
        ## IDS
        ids_en = list(df_en['id'].values)
        ids_de = list(range(1,len(df_de)+1))
        ids = np.array(ids_en + ids_de)
        
        ## TWEETS
        tweets_en = list(df_en['tweet'].values)
        tweets_de = list(df_de[0].values)
        tweets = np.array(tweets_en + tweets_de)
        tweets = process_tweets(tweets)

        # Process tweets
        label_a_en = list(df_en['subtask_a'].values)
        label_a_de = list(df_de[1].values)
        label_a = np.array(label_a_en + label_a_de)

        nums = len(df_en) + len(df_de)


        label_b = None
        label_c = None

    elif data=='train_en_test_de':
        df = pd.read_csv(filepath, sep='\t', keep_default_na=False)

        ids = np.array(df['id'].values)
        tweets = np.array(df['tweet'].values)

        # Process tweets
        tweets = process_tweets(tweets)
        label_a = np.array(df['subtask_a'].values)
        label_b = df['subtask_b'].values
        label_c = np.array(df['subtask_c'].values)

        nums = len(df)
    elif data=='train_de_test_en':
        df = pd.read_csv(filepath, sep='\t', keep_default_na=False, header=None)

        ids = np.array(range(1,len(df)+1))

        tweets = np.array(df[0].values)
        tweets = process_tweets(tweets)

        label_a = np.array(df[1].values)
        label_b = df[2].values
        label_c = None

        nums = len(df)
    elif data=='train_ende_test_de':
           ## ENG
        filepath_en = os.path.join(DATASET_PATH['en'], DATASET_DICT['en']) 
        df_en = pd.read_csv(filepath_en, sep='\t', keep_default_na=False)

        ## DE
        filepath_de = os.path.join(DATASET_PATH['de'], DATASET_DICT['de'])
        df_de = pd.read_csv(filepath_de, sep='\t', keep_default_na=False, header=None)
        
        ## IDS
        ids_en = list(df_en['id'].values)
        ids_de = list(range(1,len(df_de)+1))
        ids = np.array(ids_en + ids_de)
        
        ## TWEETS
        tweets_en = list(df_en['tweet'].values)
        tweets_de = list(df_de[0].values)
        tweets = np.array(tweets_en + tweets_de)
        tweets = process_tweets(tweets)

        # Process tweets
        label_a_en = list(df_en['subtask_a'].values)
        label_a_de = list(df_de[1].values)
        label_a = np.array(label_a_en + label_a_de)

        nums = len(df_en) + len(df_de)


        label_b = None
        label_c = None       
    else :
        raise Exception(f'Unexpected dataset, data : {data}')

    return nums, ids, tweets, label_a, label_b, label_c

def read_test_file(task, tokenizer, truncate=512, data='en'):
    if data == 'train_en_test_en':
        df1 = pd.read_csv(os.path.join(DATASET_PATH[data], 'testset-level' + task + '.tsv'), sep='\t')
        df2 = pd.read_csv(os.path.join(DATASET_PATH[data], 'labels-level' + task + '.csv'), sep=',')
        ids = np.array(df1['id'].values)
        tweets = np.array(df1['tweet'].values)
        labels = np.array(df2['label'].values)
        nums = len(df1)

    elif data == 'train_de_test_de':
        data_path = os.path.join(DATASET_PATH[data], 'germeval2018.test.txt')
        df1 = pd.read_csv(data_path, sep='\t', keep_default_na=False, header=None)
        ids = np.array(range(1,len(df1)+1))
        tweets = np.array(df1[0].values)
        labels = np.array(df1[1].values)
        nums = len(df1)

    elif data=='train_ende_test_ende':
        ## ENG 
        df_en = pd.read_csv(os.path.join(DATASET_PATH['en'], 'testset-level' + task + '.tsv'), sep='\t')
        df2 = pd.read_csv(os.path.join(DATASET_PATH['en'], 'labels-level' + task + '.csv'), sep=',')
        ## DE 
        filepath_de = os.path.join(DATASET_PATH['de'], DATASET_DICT['de_ts'])
        df_de = pd.read_csv(filepath_de, sep='\t', keep_default_na=False, header=None)


        tweets_en = list(df_en['tweet'].values)
        tweets_de = list(df_de[0].values)
        tweets = np.array(tweets_en+ tweets_de)

        ids_en = list(df_en['id'].values)
        ids_de = list(range(1,len(df_de)+1))
        ids = np.array(ids_en + ids_de)

        # Process tweets
        label_a_en = list(df2['label'].values)
        label_a_de = list(df_de[1].values)
        labels = np.array(label_a_en + label_a_de)

        nums = len(df_en) + len(df_de)

    elif data=='train_en_test_de':
        data_path = os.path.join(DATASET_PATH[data], 'germeval2018.test.txt')
        df1 = pd.read_csv(data_path, sep='\t', keep_default_na=False, header=None)
        ids = np.array(range(1,len(df1)+1))
        tweets = np.array(df1[0].values)
        labels = np.array(df1[1].values)
        nums = len(df1)

    elif data=='train_de_test_en':
        df1 = pd.read_csv(os.path.join(DATASET_PATH[data], 'testset-level' + task + '.tsv'), sep='\t')
        df2 = pd.read_csv(os.path.join(DATASET_PATH[data], 'labels-level' + task + '.csv'), sep=',')
        ids = np.array(df1['id'].values)
        tweets = np.array(df1['tweet'].values)
        labels = np.array(df2['label'].values)
        nums = len(df1)
    
    elif data=='train_ende_test_en':
        ## ENG 
        df_en = pd.read_csv(os.path.join(DATASET_PATH['en'], 'testset-level' + task + '.tsv'), sep='\t')
        df2 = pd.read_csv(os.path.join(DATASET_PATH['en'], 'labels-level' + task + '.csv'), sep=',')
        ## DE 
        filepath_de = os.path.join(DATASET_PATH['de'], DATASET_DICT['de_ts'])
        df_de = pd.read_csv(filepath_de, sep='\t', keep_default_na=False, header=None)


        tweets_en = list(df_en['tweet'].values)
        # tweets_de = list(df_de[0].values)
        # tweets = np.array(tweets_en+ tweets_de)
        tweets = np.array(tweets_en)

        ids_en = list(df_en['id'].values)
        # ids_de = list(range(1,len(df_de)+1))
        # ids = np.array(ids_en + ids_de)
        ids = np.array(ids_en)

        # Process tweets
        label_a_en = list(df2['label'].values)
        # label_a_de = list(df_de[1].values)
        # labels = np.array(label_a_en + label_a_de)
        labels = np.array(label_a_en)

        nums = len(df_en)

    elif data=='train_ende_test_de':
       ## ENG 
        df_en = pd.read_csv(os.path.join(DATASET_PATH['en'], 'testset-level' + task + '.tsv'), sep='\t')
        df2 = pd.read_csv(os.path.join(DATASET_PATH['en'], 'labels-level' + task + '.csv'), sep=',')
        ## DE 
        filepath_de = os.path.join(DATASET_PATH['de'], DATASET_DICT['de_ts'])
        df_de = pd.read_csv(filepath_de, sep='\t', keep_default_na=False, header=None)


        # tweets_en = list(df_en['tweet'].values)
        tweets_de = list(df_de[0].values)
        # tweets = np.array(tweets_en+ tweets_de)
        tweets = np.array(tweets_de)

        # ids_en = list(df_en['id'].values)
        ids_de = list(range(1,len(df_de)+1))
        # ids = np.array(ids_en + ids_de)
        ids = np.array(ids_de)

        # Process tweets
        # label_a_en = list(df2['label'].values)
        label_a_de = list(df_de[1].values)
        # labels = np.array(label_a_en + label_a_de)
        labels = np.array(label_a_de)

        nums = len(df_en)

    else:
        raise Exception(f'Unexpected dataset, data : {data}')


    # Process tweets
    tweets = process_tweets(tweets)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return ids, token_ids, lens, mask, labels


def process_tweets(tweets):
    # Process tweets
    tweets = emoji2word(tweets)
    tweets = remove_usernames(tweets)
    tweets = replace_rare_words(tweets)
    tweets = remove_replicates(tweets)
    tweets = segment_hashtag(tweets)
    tweets = remove_useless_punctuation(tweets)
    tweets = np.array(tweets)
    return tweets

def remove_usernames(sents):
    for i, sent in enumerate(sents):
        sents[i] = re.sub('@[^\s]+','@USER',sent)
    return sents

def emoji2word(sents):
    return [emoji.demojize(sent) for sent in sents]

def remove_useless_punctuation(sents):
    for i, sent in enumerate(sents):
        sent = sent.replace(':', ' ')
        sent = sent.replace('_', ' ')
        sent = sent.replace('...', ' ')
        sent = sent.replace('..', ' ')
        sent = sent.replace('’', '')
        sent = sent.replace('"', '')
        sent = sent.replace(',', '')
        sents[i] = sent
    return sents

def remove_replicates(sents):
    # if there are multiple `@USER` tokens in a tweet, replace it with `@USERS`
    # because some tweets contain so many `@USER` which may cause redundant
    for i, sent in enumerate(sents):
        if sent.find('@USER') != sent.rfind('@USER'):
            sents[i] = sent.replace('@USER ', '')
            sents[i] = '@USERS ' + sents[i]
    return sents

def replace_rare_words(sents):
    rare_words = {
        'URL': 'http'
    }
    for i, sent in enumerate(sents):
        for w in rare_words.keys():
            sents[i] = sent.replace(w, rare_words[w])
    return sents

def segment_hashtag(sents):
    # E.g. '#LunaticLeft' => 'lunatic left'
    for i, sent in enumerate(sents):
        sent_tokens = sent.split(' ')
        for j, t in enumerate(sent_tokens):
            if t.find('#') == 0:
                sent_tokens[j] = ' '.join(wordsegment.segment(t))
        sents[i] = ' '.join(sent_tokens)
    return sents


def task_a(filepath: str, tokenizer, truncate=512, data='en'):
    nums, ids, tweets, label_a, _, _ = read_file(filepath, data=data)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return ids, token_ids, lens, mask, label_a
