
def get_data_loader(train_data, test_data):
    train_dataset = InteractionDataset(train_data)
    test_dataset = InteractionDataset(test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=True)

    return train_loader, test_loader


def write_result(probas1, y, test_loss, test_acc, test_auc, roc_auc, resultFolder):
    with open(os.path.join(resultFolder, 'probas1-HNet-DNN.txt'), 'w') as f:
        for val in probas1:
            f.write('{}\n'.format(val))

    with open(os.path.join(resultFolder, 'y-HNet-DNN.txt'), 'w') as f:
        for val in y:
            f.write('{}\n'.format(val))

    with open(os.path.join(resultFolder, 'metrics.txt'), 'w') as f:
        f.write('Test loss {:.4}, acc {:.4}, auc {:.4}\n'.format(test_loss, test_acc, test_auc))
        f.write('Test auc {:.4}\n'.format(roc_auc))


def get_data_split(original_data, expanded_data, negative_data, setting_mode):
    n_org = len(original_data)
    n_exp = len(expanded_data)
    np.random.shuffle(original_data)
    np.random.shuffle(expanded_data)
    train_neg_data = negative_data[:n_org - n_exp]
    test_neg_data = negative_data[n_org - n_exp:n_org]
    if setting_mode == 1:
        all_pos_data = original_data + expanded_data
        np.random.shuffle(all_pos_data)
        train_pos_data = all_pos_data[:n_org - n_exp] 
        test_pos_data = all_pos_data[n_org - n_exp:n_org]
    elif setting_mode == 2:
        train_pos_data = original_data[:n_org - n_exp]
        test_pos_data = original_data[n_org - n_exp:n_org]
    else:
        train_pos_data = original_data[:n_org - n_exp]
        test_pos_data = expanded_data[:n_exp]
    train_data = train_pos_data + train_neg_data
    test_data = test_pos_data + test_neg_data
    return train_data, test_data


def get_positive_pairs(fname):
    original_pairs = []
    expanded_pairs = []
    with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            segs = line.strip().split()
            drug_id, disease_id = segs[0], segs[1]
            if segs[2] == 'original':
                original_pairs.append((drug_id, disease_id, 1))
            else:
                expanded_pairs.append((drug_id, disease_id, 1))
    return original_pairs, expanded_pairs


def get_negative_pairs(fname='all.txt'):
    negative_pairs = []
    with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
        for line in f:
            drug_id, disease_id, label = line.strip().split()
            if int(label) == 0:
                negative_pairs.append((drug_id, disease_id, 0))
    return negative_pairs


def main():
    data_file = 'DiDrMat_{}.tsv'.format(args.data)
    original_pairs, expanded_pairs = get_positive_pairs(data_file)
    negative_pairs = get_negative_pairs()
    print('File: {}'.format(data_file))
    for setting_mode in [1, 2, 3]:
        print('setting mode', setting_mode)
        train_data, test_data = get_data_split(original_pairs, expanded_pairs, negative_pairs, setting_mode)
        train_loader, test_loader = get_data_loader(train_data, test_data)
        
        resultFolder = os.path.join(folder, '{}_setting{}_result'.format(args.data, setting_mode))
        if not os.path.exists(resultFolder):
            print('mkdir {}'.format(resultFolder))
            os.mkdir(resultFolder)

        best_model = 'best_DNN_model_{}_setting{}.pkl'.format(args.data, setting_mode)
        if not os.path.exists(os.path.join(folder, best_model)):
            fc_dnn = generate_model(train_loader)
            save_model(fc_dnn, os.path.join(folder, best_model))
        else:
            fc_dnn = load_model(os.path.join(folder, best_model))

        test_loss, test_acc, test_auc = classification_accuracy(fc_dnn, test_loader)
        print('Test loss {:.4}, acc {:.4}, auc {:.4}'.format(test_loss, test_acc, test_auc))
        probas1, y = get_result_from_model(fc_dnn, test_loader)
        average_precision = average_precision_score(y, probas1)
        print('Test aupr {:.4}'.format(average_precision))
        fpr, tpr, thresholds = roc_curve(y, probas1, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print('Test auc {:.4}'.format(roc_auc))
        write_result(probas1, y, test_loss, test_acc, test_auc, roc_auc, resultFolder)
    print('Done')


if __name__ == '__main__':
    main()
