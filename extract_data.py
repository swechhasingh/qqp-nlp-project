import numpy as np
import pandas as pd

from utils import create_vocab, tokenize_data
from sklearn.model_selection import train_test_split


def extract_data(file_path='./data/train.csv'):
    train_df = pd.read_csv(file_path, header=0)
    train_df = train_df.drop([363362, 105780, 201841])

    X_df = train_df.drop(columns=['is_duplicate'])
    Y_df = train_df['is_duplicate']

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, Y_df, test_size=0.2, random_state=42, stratify=Y_df)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    n_train = len(X_train)
    #  qid1, qid2 = df['qid1'].to_numpy(), df['qid2'].to_numpy()
    # q1s, q2s = df['question1'].to_numpy(), df['question2'].to_numpy()
    # labels = df['is_duplicate']
    print("Train set: No. of question pairs: {}".format(n_train))
    train_data = {'q1s': X_train['question1'].to_numpy(),
                  'q2s': X_train['question2'].to_numpy(),
                  'labels': y_train.to_numpy()}

    n_val = len(X_val)
    print("Validation set: No. of question pairs: {}".format(n_val))
    val_data = {'q1s': X_val['question1'].to_numpy(),
                'q2s': X_val['question2'].to_numpy(),
                'labels': y_val.to_numpy()}

    n_test = len(X_test)
    print("Test set: No. of question pairs: {}".format(n_test))
    test_data = {'q1s': X_test['question1'].to_numpy(),
                 'q2s': X_test['question2'].to_numpy(),
                 'labels': y_test.to_numpy()}
    return train_data, val_data, test_data


def load_and_split_data(file_path='./data/train.csv'):
    df = pd.read_csv(file_path, header=0)
    df = df.drop([363362, 105780, 201841])
    qid1, qid2 = df['qid1'].to_numpy(), df['qid2'].to_numpy()
    q1s, q2s = df['question1'].to_numpy(), df['question2'].to_numpy()
    labels = df['is_duplicate'].to_numpy()

    n_pairs = len(labels)
    para_idx = np.nonzero(labels)[0]
    non_para_idx = np.nonzero(1 - labels)[0]
    n_paraphrases = len(labels[labels == 1])
    n_non_para = len(labels[labels == 0])
    print('Number of paraphrases out of {} pairs: {}'.format(n_pairs, n_paraphrases))
    print('Number of non paraphrases out of {} pairs: {}'.format(n_pairs, n_non_para))

    print('Paraphrase samples indices: {}'.format(para_idx))
    print('Non paraphrase samples indices: {}'.format(non_para_idx))

    np.random.seed(seed=12)
    para_idx = np.random.permutation(para_idx)
    non_para_idx = np.random.permutation(non_para_idx)

    val_para_idx = para_idx[:5000].tolist()
    test_para_idx = para_idx[5000:10000].tolist()
    train_para_idx = para_idx[10000:].tolist()

    val_nonpara_idx = non_para_idx[:5000].tolist()
    test_nonpara_idx = non_para_idx[5000:10000].tolist()
    train_nonpara_idx = non_para_idx[10000:].tolist()

    train_idx = train_para_idx + train_nonpara_idx
    val_idx = val_para_idx + val_nonpara_idx
    test_idx = test_para_idx + test_nonpara_idx

    train_q1s = q1s[train_idx]
    train_q2s = q2s[train_idx]
    train_labels = labels[train_idx]
    n_train = len(train_labels)
    print("Train set: No. of question pairs: {}".format(n_train))
    train_data = {'q1s': train_q1s,
                  'q2s': train_q2s,
                  'labels': train_labels}

    val_q1s = q1s[val_idx]
    val_q2s = q2s[val_idx]
    val_labels = labels[val_idx]
    n_val = len(val_labels)
    print("Validation set: No. of question pairs: {}".format(n_val))
    val_data = {'q1s': val_q1s,
                'q2s': val_q2s,
                'labels': val_labels}

    test_q1s = q1s[test_idx]
    test_q2s = q2s[test_idx]
    test_labels = labels[test_idx]
    n_test = len(test_labels)
    print("Test set: No. of question pairs: {}".format(n_test))
    test_data = {'q1s': test_q1s,
                 'q2s': test_q2s,
                 'labels': test_labels}
    return train_data, val_data, test_data


def main():
    # load_and_split_data(file_path='./data/train.csv')
    train_data, val_data, test_data = extract_data(file_path='./data/train.csv')
    print('\n')

if __name__ == '__main__':
    main()
