OLID_PATH = './OLIDv1.0'
GERMEVAL_PATH = './GermEval2018'
PERSIAN_PATH  = './Persian'
SAVE_PATH = './save'
LABEL_DICT_OLID = {
    'a': {'OFF': 0, 'NOT': 1},
    'b': {'TIN': 0, 'UNT': 1, 'NULL': 2},
    'c': {'IND': 0, 'GRP': 1, 'OTH': 2, 'NULL': 3}
}
LABEL_DICT_GERMEVAL = {
    'a': {'OFFENSE': 0, 'OTHER': 1},
    'b': {'OTHER': 0, 'INSULT': 1, 'PROFANITY': 2, 'ABUSE': 3},
}
LABEL_DICT_EN_DE = {
    'a': {'OFFENSE': 0, 'OTHER': 1, 'OFF': 0, 'NOT': 1},
}
LABEL_DICT_FA = {
    'a': {'OFF': 0, 'NOT': 1, 'RELIG': 2, 'NATIO': 3, 'RACE': 4},
}