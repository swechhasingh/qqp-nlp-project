import numpy as np
import time
import torch
import argparse
from utils import *
from extract_data import load_and_split_data, extract_data
from dataset import QQPDataset
from models.bimpm_model import BiMPM
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle


def train(epoch, data_loader, model, optimizer):
    model.train()
    total_loss = 0.
    start_time = time.time()
    count = 0.

    for i, batch in enumerate(data_loader):
        q1s = batch['q1']
        q2s = batch['q2']
        q1_len = batch['q1_len']
        q2_len = batch['q2_len']
        true_labels = batch['label'].to(dtype=torch.float)

        optimizer.zero_grad()

        output_logits = model(q1s, q2s, q1_len, q2_len)
        loss = model.criterion(output_logits, true_labels)

        pred_labels = (torch.sigmoid(output_logits) > 0.5).to(dtype=torch.float)
        count += torch.sum(pred_labels ==
                           true_labels).to(dtype=torch.float).item() / len(true_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'loss {}'.format(epoch, (i + 1), len(data_loader),
                                   total_loss / (i + 1)))
    accuracy = 100 * count / len(data_loader)
    avg_loss = total_loss / len(data_loader)
    print('|Epoch {:3d} | accuracy {}'.format(epoch, accuracy))
    return avg_loss, accuracy


def evaluate(eval_model, data_loader):
    eval_model.eval()
    total_loss = 0.
    count = 0.
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            q1s = batch['q1']
            q2s = batch['q2']
            q1_len = batch['q1_len']
            q2_len = batch['q2_len']
            true_labels = batch['label'].to(dtype=torch.float)

            output_logits = eval_model(q1s, q2s, q1_len, q2_len)
            loss = eval_model.criterion(output_logits, true_labels)

            total_loss += loss.item()

            pred_labels = (torch.sigmoid(output_logits) > 0.5).to(dtype=torch.float)
            count += torch.sum(pred_labels ==
                               true_labels).to(dtype=torch.float).item() / len(true_labels)
            y_true.append(true_labels.detach().cpu().numpy())
            y_pred.append(pred_labels.detach().cpu().numpy())
    accuracy = 100 * count / len(data_loader)
    avg_loss = total_loss / len(data_loader)

    return avg_loss, accuracy, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', default=20, type=int, help='no. of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--cos_sim_dim', default=20, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--is_pretrained', action='store_true')
    parser.add_argument('--embed_dim', default=300,
                        type=int, help='embedding dimension')
    args = parser.parse_args()
    print(args)
    # load_and_split_data(file_path='./data/train.csv')
    train_data, val_data, test_data = extract_data(file_path='./data/train.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs = args.max_epochs  # The number of epochs
    best_model = None
    best_epoch = 0

    train_set = QQPDataset(train_data, split='train', vocab=None,
                           word2idx=None, pre_process=None, device=device, debug=args.debug)
    val_set = QQPDataset(val_data, split='val', vocab=train_set.vocab,
                         word2idx=train_set.word2idx, pre_process=None, device=device, debug=args.debug)

    test_set = QQPDataset(test_data, split='test', vocab=train_set.vocab,
                          word2idx=train_set.word2idx, pre_process=None, device=device, debug=args.debug)

    # use only first time for loading GloVe
    # parse_n_store_glove(file_path='./data')

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    vocab_size = len(train_set.vocab)
    K = args.cos_sim_dim

    if args.is_pretrained:
        # use only once
        word2GloVe = create_word2GloVe_dict(file_path='./data')

        word2GloVe = pickle.load(open(f'./data/word2GloVe_dict.pkl', 'rb'))
        # generate GloVe embeddings for words in vocabulary
        vocab_embeddings = get_glove_embeddings(train_set.vocab, word2GloVe)
        vocab_embeddings = torch.from_numpy(vocab_embeddings).to(device)

        model = BiMPM(vocab_size, embed_dim=None,
                      weight_matrix=vocab_embeddings, hidden_size=100, K=K)
    else:
        embed_dim = args.embed_dim
        model = BiMPM(vocab_size, embed_dim, weight_matrix=None, hidden_size=100, K=K)

    model.to(device)

    lr = args.lr  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1,
    # patience=3, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # for i_batch, sample_batched in enumerate(dataloader):
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    best_model = None
    k = 0
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train(epoch, train_dataloader, model, optimizer)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_dataloader)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '.format(epoch,
                                                                                (time.time(
                                                                                ) - epoch_start_time),
                                                                                val_loss))
        print('|Epoch {:3d} | valid accuracy {}'.format(epoch, val_acc))
        print('-' * 89)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model = model
            best_epoch = epoch
            torch.save(model.state_dict(), "./best_model_lstm.pt")
            pickle.dump(y_true, open(f'best_val_true_labels.pkl', 'wb'))
            pickle.dump(y_pred, open(f'best_val_pred_labels.pkl', 'wb'))

            k = 0
        elif k < args.patience:
            k += 1
        else:
            break
        # scheduler.step()
    print('Best val loss: {} | acc: {} at epoch {}'.format(
        best_val_loss, best_val_acc, best_epoch))
    test_loss, test_acc, y_true, y_pred = evaluate(best_model, test_dataloader)
    print('Test | loss: {} | acc: {}'.format(
        test_loss, test_acc))
    pickle.dump(y_true, open(f'test_true_labels.pkl', 'wb'))
    pickle.dump(y_pred, open(f'test_pred_labels.pkl', 'wb'))

    log_results = {'train_acc': train_accs,
                   'train_loss': train_losses,
                   'val_acc': val_accs,
                   'val_loss': val_losses,
                   'best_loss': best_val_loss,
                   'best_acc': best_val_acc,
                   'test_loss': test_loss,
                   'test_acc': test_acc
                   }
    pickle.dump(log_results, open(f'log_results.pkl', 'wb'))

if __name__ == '__main__':
    main()
