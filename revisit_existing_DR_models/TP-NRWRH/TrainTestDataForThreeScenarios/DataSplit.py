import numpy as np
folder='C:/Users/'

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

#Transform to train_data, test_data to same format of "DiDrAMat" matrix   
Drug_list = [sub.replace("\n", "") for sub in list(open('DrugsName', 'r', encoding='utf-8'))]
Dis_list = [sub.replace("\n", "") for sub in list(open('DiseasesName', 'r', encoding='utf-8'))]

SPLIT_SETTINGS = ['IC2C09', 'MC2C07']
SETTING_NUMBS = [1, 2, 3]

filenames = {'IC2C09' : 'DiDrMat_ICD10_cluster2_complete_th0.9.tsv',
            'MC2C07' : 'DiDrMat_MeSH_cluster2_complete_th0.7.tsv'}

for SPLIT_SETTING in SPLIT_SETTINGS:
    for SETTING_NUMB in SETTING_NUMBS:
        data_file = filenames[SPLIT_SETTING]
        original_pairs, expanded_pairs = get_positive_pairs(data_file)
        negative_pairs = get_negative_pairs()
        train_data, test_data = get_data_split(original_pairs, expanded_pairs, negative_pairs, setting_mode=SETTING_NUMB)
        print("train_data,", len(train_data), '; test_data,', len(test_data))

        matrix_train = np.zeros(shape=(len(Dis_list), len(Drug_list)), dtype=int)
        for t_data in train_data:
            if t_data[2]==1:        
                i = Dis_list.index(t_data[1])
                j = Drug_list.index(t_data[0])
                matrix_train[i][j] = 1

        matrix_test = np.zeros(shape=(len(Dis_list), len(Drug_list)), dtype=int)
        for t_data in test_data:
            if t_data[2]==1:        
                i = Dis_list.index(t_data[1])
                j = Drug_list.index(t_data[0])
                matrix_test[i][j] = 1

        save_dir = 'C:/Users/'
        save_dir = save_dir + f'/{SPLIT_SETTING + str(SETTING_NUMB)}'

        import pandas as pd
        df_train = pd.DataFrame(matrix_train)
        df_train.to_csv(save_dir + '/Train/DiDrAMat_tr', index=False)

        df_test = pd.DataFrame(matrix_test)
        df_test.to_csv(save_dir + '/Test/DiDrAMat_te', index=False)