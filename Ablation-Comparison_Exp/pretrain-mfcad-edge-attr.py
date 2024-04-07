from __future__ import division
from __future__ import print_function
import os
import time
import h5py
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from models import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dataloader(file_path):
    hf = h5py.File(file_path, 'r')

    for key in list(hf.keys()):
        group = hf.get(key)
        v_1 = np.array(group.get("V_1"))
        a_1_idx = np.array(group.get("A_1_idx"))
        v_2 = np.array(group.get("V_2"))
        labels = np.array(group.get("labels"))

        yield v_1, a_1_idx, v_2, labels

    hf.close()


def load_raw_data(node_feature, adj, edge_feature, labels):
    node_feature = torch.tensor(node_feature, dtype=torch.float).to(device)
    adj = np.array([adj[:, 0], adj[:, 1]])
    adj = torch.tensor(adj, dtype=torch.long).to(device)
    
    edge_feature = np.delete(edge_feature, 1, axis=1)
    edge_feature = torch.tensor(edge_feature, dtype=torch.float).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    data_loader = Data(x=node_feature, edge_index=adj, edge_attr=edge_feature, y=labels)

    return data_loader


def pretrain_epoch(data):
    encoder.train()
    classifier.train()
    optimizer_encoder.zero_grad()
    embeddings = encoder(data.x, data.edge_index, data.edge_attr).to(device)
    output = classifier(embeddings).to(device)
    output = F.log_softmax(output, dim=1)
    labels = data.y
    loss_train = F.nll_loss(output, labels)
    loss_train.backward()
    optimizer_encoder.step()
    optimizer_classifier.step()
    if torch.cuda.is_available():
        output = output.cpu().detach()
        labels = labels.cpu().detach()
    acc_train = accuracy(output, labels)
    f1_train = f1(output, labels)

    return acc_train, f1_train


def pretest_epoch(data):
    encoder.eval()
    classifier.eval()
    embeddings = encoder(data.x, data.edge_index, data.edge_attr).to(device)
    output = classifier(embeddings).to(device)
    output = F.log_softmax(output, dim=1)
    labels = data.y
    if torch.cuda.is_available():
        output = output.cpu().detach()
        labels = labels.cpu().detach()
    acc_test = accuracy(output, labels)
    f1_test = f1(output, labels)

    return acc_test, f1_test


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--num_layers', type=int, default=13, help='Number of layers.')
    parser.add_argument('--fea_vec_dim', type=int, default=7, help='Dimension of node features.')
    parser.add_argument('--edge_vec_dim', type=int, default=2, help='Dimension of edge features.')
    parser.add_argument('--dataset', default='mfcad++', help='Dataset:sheet_metal/mfcad++')
    parser.add_argument('--pretrain_model', required=False, help='Existing model path.')
    parser.add_argument('--overwrite_pretrain', action='store_true', help='Delete existing pre-train model')
    parser.add_argument('--output_path', default='./smnet_pretrain_model', help='Path for output pre-trained model.')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_tmp = os.path.join(args.output_path, str(args.dataset))
    if args.overwrite_pretrain and os.path.exists(path_tmp):
        cmd = "rm -rf " + path_tmp
        os.system(cmd)
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    seed_everything(args.seed)
    train_set_path = "autodl-tmp/train_MFCAD++_batch.h5"
    val_set_path = "autodl-tmp/val_MFCAD++_batch.h5"
    test_set_path = "autodl-tmp/test_MFCAD++_batch.h5"
    pretrain_class = 30

    # Model and optimizer
    encoder = SMNetEncoder(in_channels=args.fea_vec_dim, hidden_channels=args.hidden,
                           num_layers=args.num_layers, edge_vec_dim=args.edge_vec_dim).to(device)
    classifier = Classifier(hidden_channels=args.hidden, class_num=pretrain_class).to(device)
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.pretrain_model:
        checkpoint = torch.load(args.pretrain_model)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
        optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
        epoch = checkpoint['epoch']

    # Train model
    t_total = time.time()
    pre_train_acc = []
    best_dev_acc = 0.
    tolerate = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        print("-------Epochs {}-------".format(epoch))
        generator1 = dataloader(train_set_path)
        generator2 = dataloader(val_set_path)
        for i, (feature_vec_1, adj_1, edge_feature_1, face_index_1) in enumerate(generator1):
            data_load = load_raw_data(feature_vec_1, adj_1, edge_feature_1, face_index_1)
            acc_training, f1_training = pretrain_epoch(data_load)
            pre_train_acc.append(acc_training)

        # validation
        pre_dev_acc = []
        pre_dev_f1 = []
        for j, (feature_vec_2, adj_2, edge_feature_2, face_index_2) in enumerate(generator2):
            data_load = load_raw_data(feature_vec_2, adj_2, edge_feature_2, face_index_2)
            acc_testing, f1_testing = pretest_epoch(data_load)
            pre_dev_acc.append(acc_testing)
            pre_dev_f1.append(f1_testing)

        curr_dev_acc = np.array(pre_dev_acc).mean(axis=0)
        print("Pre-Train_Accuracy: {}".format(np.array(pre_train_acc).mean(axis=0)))
        print("Pre-Valid_Accuracy: {}, Pre-Valid_F1: {}".format(curr_dev_acc, np.array(pre_dev_f1).mean(axis=0)))
        if curr_dev_acc >= best_dev_acc:
            best_dev_acc = curr_dev_acc
            save_path = os.path.join(args.output_path, args.dataset, str(args.seed) + "_" + (str(epoch) + ".pth"))
            tolerate = 0
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                'optimizer_classifier_state_dict': optimizer_classifier.state_dict(),
            }, save_path)
            print("model saved at " + save_path)
            best_epoch = epoch
        else:
            continue
    print("Best pretrain epoch: " + str(best_epoch))

    # final test
    best_pretrain_path = os.path.join(args.output_path, args.dataset, str(args.seed) + "_" + str(best_epoch) + ".pth")
    final_model = torch.load(best_pretrain_path)
    encoder.load_state_dict(final_model["encoder_state_dict"])
    classifier.load_state_dict(final_model['classifier_state_dict'])
    pre_test_acc = []
    pre_test_f1 = []
    generator3 = dataloader(test_set_path)
    for k, (feature_vec_3, adj_3, edge_feature_3, face_index_3) in enumerate(generator3):
        data_load = load_raw_data(feature_vec_3, adj_3, edge_feature_3, face_index_3)
        acc_final_test, f1_final_test = pretest_epoch(data_load)
        pre_test_acc.append(acc_final_test)
        pre_test_f1.append(f1_final_test)
    print("Pre-Test_Accuracy: {}, Pre-Test_F1: {}".format(np.array(pre_test_acc).mean(axis=0),
                                                          np.array(pre_test_f1).mean(axis=0)))

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
