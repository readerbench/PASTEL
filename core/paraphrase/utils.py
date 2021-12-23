import os
import random
import numpy as np
import torch
from rb.core.document import Document
from keras.preprocessing import text


DATA_FOLDER = "../../data/"
RESULTS_FOLDER = "../../data/results/"


def get_lemma(word):
    if word.lemma != "-PRON-":
        return word.lemma
    return word.text


def get_pos(d: Document):
    return set([get_lemma(word) for word in d.get_words() if word.is_alpha])


def get_words(d: Document):
    return set([get_lemma(word) for word in d.get_words() if word.is_alpha])


def filter_example(source_doc, prod_doc, data_line):
    s_cont = get_words(source_doc)
    p_cont = get_words(prod_doc)

    # Filter short
    if len(p_cont) == 1:
        print("Filter short")
        return True
    # Filter garbage
    if 'Garbage_content_bin' in data_line.index and data_line['Garbage_content_bin'] == 1:
        print("Filter garbage")
        return True
    # Filter irrelevant
    if 'Garbage_content_bin' in data_line.index and data_line['Irrelevant_bin'] == 1:
        print("Filter irrelevant")
        return True
    # Copy paste (same lemmas)
    if len([w for w in s_cont if w not in p_cont]) + len([w for w in p_cont if w not in s_cont]) == 1:
        print("Filter copy paste")
        return True

    return False


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_tokenizer(input_list, project_name, recompute_embedding_mat=False, generic_tokenizer=False):
    GLOVE_EMBEDDING_PATH = f'{DATA_FOLDER}glove.6B.300d.txt'
    if generic_tokenizer:
        project_name="ALL"
    if recompute_embedding_mat:
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(input_list + ["unknown"])
        glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, emb_path=GLOVE_EMBEDDING_PATH, unknown_token='unknown')
        print('n unknown words (glove): ', len(unknown_words_glove))
        glove_matrix = torch.Tensor(glove_matrix)
        torch.save(glove_matrix, f"{DATA_FOLDER}{project_name}_glove_subset.obj")
        torch.save(tokenizer, f"{DATA_FOLDER}{project_name}_tokenizer.obj")
    else:
        glove_matrix = torch.load(f"{DATA_FOLDER}{project_name}_glove_subset.obj")
        tokenizer = torch.load(f"{DATA_FOLDER}{project_name}_tokenizer.obj")
    return tokenizer, glove_matrix

def build_matrix(word_index, emb_path, unknown_token='unknown'):
    matrix = [[] for _ in word_index] + [[]]
    word_dict = {word_index[word]: word for id, word in enumerate(word_index)}
    with open(emb_path, 'r') as fp:
        i = 0
        j = 0
        line = fp.readline()
        while line:
            toks = line.split(" ")
            word = toks[0]
            embedding = [float(x) for x in toks[1:]]
            if word in word_index:
                matrix[word_index[word]] = embedding
                j += 1
            if f"{word}'s" in word_index:
                matrix[word_index[f"{word}'s"]] = embedding
                j += 1
            if (word.endswith('s') or word.endswith('\'')) and word[:-1] in word_index and len(matrix[word_index[word[:-1]]]) == 0:
                matrix[word_index[word[:-1]]] = embedding
                j += 1
            if i % 100 == 0:
                print(i, j)
            i += 1
            line = fp.readline()

    unknown_words = []
    for i in range(1, len(matrix)):
        if len(matrix[i]) == 0:
            matrix[i] = matrix[word_index[unknown_token]]
            unknown_words.append(word_dict[i])
    matrix[0] = matrix[word_index[unknown_token]]
    print(unknown_words)
    print(len(unknown_words))

    return matrix, unknown_words
