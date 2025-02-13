from ast import arg
import os
from unittest.util import _MAX_LENGTH
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import task_a, read_test_file, make_dict
from config import *
from cli import get_args
from utils import load, save_tokenizer
from datasets import HuggingfaceDataset, ImbalancedDatasetSampler, LABEL_DICT
from models.bert import BERT, RoBERTa, XLM_RoBERTa, MultilingualBERT, GE_BERT, ParsBERT, BERTTWEET_FA, MiniBert
from transformers import BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, get_cosine_schedule_with_warmup, AutoTokenizer
from trainer import Trainer

DATASET_DICT = {
    'train_en_test_en': 'olid-training-v1.0.tsv',
    'train_en_test_de': 'olid-training-v1.0.tsv',
    'train_de_test_de': 'germeval2018.training.txt',
    'train_de_test_en': 'germeval2018.training.txt',
    'train_fa_test_fa': 'persian_train.xlsx',
    'train_ende_test_en': "dummy",
    'train_ende_test_de': "dummy",
    'train_ende_test_ende': "dummy"
}
DATASET_PATH = {
    'train_en_test_en': OLID_PATH,
    'train_en_test_de': OLID_PATH,
    'train_de_test_de': GERMEVAL_PATH,
    'train_de_test_en': GERMEVAL_PATH,
    'train_fa_test_fa': PERSIAN_PATH,
    'train_ende_test_en': "dummy",
    'train_ende_test_de': "dummy",
    'train_ende_test_ende': "dummy"
}

if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    task = args['task']
    model_name = args['model']
    model_size = args['model_size']
    truncate = args['truncate']
    epochs = args['epochs']
    lr = args['learning_rate']
    wd = args['weight_decay']
    bs = args['batch_size']
    patience = args['patience']
    dataset = args['dataset']

    TRAIN_PATH = os.path.join(DATASET_PATH[dataset], DATASET_DICT[dataset])
    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create related dict
    make_dict(TRAIN_PATH, dataset)

    num_labels = len(LABEL_DICT[dataset][task])

    # Set tokenizer for different models
    exception_message = 'Unexpected model.(deleted), model_name : {}'.format(model_name)
    if model_name == 'bert':
        if task == 'all':
            raise Exception(exception_message)
        else:
            model = BERT(model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'minibert':
        model = MiniBert(model_size, args=args, word_embedding=args['dim'], num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta':
        if task == 'all':
            raise Exception(exception_message)
        else:
            model = RoBERTa(model_size, args=args, num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')
    elif model_name == 'bert-gate' and task == 'all':
        raise Exception(exception_message)
    elif model_name == 'roberta-gate' and task == 'all':
        raise Exception(exception_message)
    elif model_name == 'xlm-roberta':
        print(f'using xlm-roberta-{model_size} model.')
        model = XLM_RoBERTa(model_size, args=args, num_labels=num_labels)
        tokenizer = XLMRobertaTokenizer.from_pretrained(f'xlm-roberta-base')
        save_tokenizer(tokenizer, './save/tokenizer')
        assert tokenizer != None
    elif model_name == 'pars-bert':
        print(f'using bert-{model_size}-parsbert-uncased.')
        model = ParsBERT(model_size, args=args, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(f'HooshvareLab/bert-base-parsbert-uncased')
        save_tokenizer(tokenizer, './save/tokenizer')
        assert tokenizer != None
    elif model_name == 'bert-tweet':
        print(f'using bert-tweet-farsi.')
        model = BERTTWEET_FA(model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained('arm-on/BERTweet-FA', model_max_length=130)
        save_tokenizer(tokenizer, './save/tokenizer')
        assert tokenizer != None
    elif model_name == 'bert-multilingual':
        print(f'using bert-{model_size}-multilingual-uncased model.')
        model = MultilingualBERT(model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-base-multilingual-uncased')
        save_tokenizer(tokenizer, './save/tokenizer')
        assert tokenizer != None
    elif model_name == 'gebert':
        print(f'using bert-{model_size}-german-dbmdz-uncased model.')
        model = GE_BERT(model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-base-german-dbmdz-uncased')
        assert tokenizer != None
    else :
        raise Exception(f'Unexpected model., model_name : {model_name}')

    # Move model to correct device
    model = model.to(device=device)

    if args['ckpt'] != '':
        model.load_state_dict(load(args['ckpt']))

    # Read in data depends on different subtasks
    if task in ['a']:
        data_methods = {'a': task_a}
        ids, token_ids, lens, mask, labels = data_methods[task](TRAIN_PATH, tokenizer=tokenizer, truncate=truncate, data=dataset)
        test_ids, test_token_ids, test_lens, test_mask, test_labels = read_test_file(task, tokenizer=tokenizer, truncate=truncate, data=dataset)
        _Dataset = HuggingfaceDataset
    # elif task in ['all']:
    #     ids, token_ids, lens, mask, label_a, label_b, label_c = all_tasks(TRAIN_PATH, tokenizer=tokenizer, truncate=truncate)
    #     test_ids, test_token_ids, test_lens, test_mask, test_label_a, test_label_b, test_label_c = read_test_file_all(tokenizer)
    #     labels = {'a': label_a, 'b': label_b, 'c': label_c}
    #     test_labels = {'a': test_label_a, 'b': test_label_b, 'c': test_label_c}
    #     _Dataset = HuggingfaceMTDataset

    datasets = {
        'train': _Dataset(
            input_ids=token_ids,
            lens=lens,
            mask=mask,
            labels=labels,
            num_labels=num_labels,
            task=task,
            data=dataset
        ),
        'test': _Dataset(
            input_ids=test_token_ids,
            lens=test_lens,
            mask=test_mask,
            labels=test_labels,
            num_labels=num_labels,
            task=task,
            data=dataset
        )
    }

    sampler = ImbalancedDatasetSampler(datasets['train']) if task in ['a', 'b', 'c'] else None
    dataloaders = {
        'train': DataLoader(
            dataset=datasets['train'],
            batch_size=bs,
            sampler=sampler
        ),
        'test': DataLoader(dataset=datasets['test'], batch_size=bs)
    }

    criterion = torch.nn.BCEWithLogitsLoss() if args['multilabel'] else torch.nn.CrossEntropyLoss()

    if args['scheduler']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        # A warmup scheduler
        t_total = epochs * len(dataloaders['train'])
        warmup_steps = np.ceil(t_total / 10.0) * 2
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = None

    trainer = Trainer(
        model=model,
        epochs=epochs,
        dataloaders=dataloaders,
        criterion=criterion,
        loss_weights=args['loss_weights'],
        clip=args['clip'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=patience,
        task_name=task,
        dataset_name=dataset,
        model_name=model_name,
        multilabel=args['multilabel'],
        seed=args['seed'],
        save_model=args['save'],
    )

    if task in TASKS:
        trainer.train()
    else:
        trainer.train_m()
