def get_data(original_positive_index, expanded_positive_index, setting_mode):
    n_org = len(original_positive_index)
    n_exp = len(expanded_positive_index)
    np.random.shuffle(original_positive_index)
    np.random.shuffle(expanded_positive_index)
    if setting_mode == 1:
        all_pos_data = original_positive_index + expanded_positive_index
        np.random.shuffle(all_pos_data)
        train_pos_data = all_pos_data[:n_org - n_exp] 
        test_data = all_pos_data[n_org - n_exp:n_org]
    elif setting_mode == 2:
        train_pos_data = original_positive_index[:n_org - n_exp]
        test_data = original_positive_index[n_org - n_exp:n_org]
    else:
        train_pos_data = original_positive_index[:n_org - n_exp]
        test_data = expanded_positive_index[:n_exp]
    train_data = train_pos_data
    return train_data, test_data
    
    
 def set_train_test_data(original_path, expanded_path):    
    print('train data path: ' + original_path + ' ' + expanded_path)
    
    #Read original and expanded tsv files
    original_R = [];expanded_R = []
    with open(original_path, 'r') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            original_R.append(row[1:])
    with open(expanded_path, 'r') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            expanded_R.append(row[1:])
    original_R = np.array(original_R); expanded_R = np.array(expanded_R)
    original_Rtensor = original_R.transpose()
    expanded_Rtensor = expanded_R.transpose()

    #Get negative data
    original_positive_index = []
    expanded_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(original_Rtensor)[0]):
        for j in range(np.shape(original_Rtensor)[1]):
            if int(original_Rtensor[i][j]) == 1:
                original_positive_index.append([i, j])
            if int(expanded_Rtensor[i][j]) == 1:
                expanded_positive_index.append([i, j])
            if int(original_Rtensor[i][j]) == 0 and int(expanded_Rtensor[i][j]) == 0:
                whole_negative_index.append([i, j])
    n_org = len(original_positive_index)
    n_exp = len(expanded_positive_index)
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                    size=1 * n_org, replace=False)
                                                    
    #Set train and test data for negative samples                                                
    train_negative_index = negative_sample_index[:n_org - n_exp]
    test_negative_index = negative_sample_index[n_org - n_exp:]
    
    return train_positive_index, train_negative_index, test_positive_index, test_negative_index, whole_negative_index


def split_train_test_for_three_scenarios(train_positive_index, train_negative_index, test_positive_index, test_negative_index, whole_negative_index):
    #indexes for three scernarios
    setting_modes = [1, 2, 3]

    #split train test datasets for three scenarios
    DTItraintest_dict = {}
    for setting_mode in setting_modes:
        train_positive_index, test_positive_index = get_data(original_positive_index, expanded_positive_index, setting_mode)
        # whole_negative_index=np.array(whole_negative_index)
        train_data_set = np.zeros((len(train_positive_index) + len(train_negative_index), 3), dtype=int)
        count = 0
        for i in train_positive_index:
            train_data_set[count][0] = i[0]
            train_data_set[count][1] = i[1]
            train_data_set[count][2] = 1
            count += 1
        for i in train_negative_index:
            train_data_set[count][0] = whole_negative_index[i][0]
            train_data_set[count][1] = whole_negative_index[i][1]
            train_data_set[count][2] = 0
            count += 1
        test_data_set = np.zeros((len(test_positive_index) + len(test_negative_index), 3), dtype=int)
        count = 0
        for i in test_positive_index:
            test_data_set[count][0] = i[0]
            test_data_set[count][1] = i[1]
            test_data_set[count][2] = 1
            count += 1
        for i in test_negative_index:
            test_data_set[count][0] = whole_negative_index[i][0]
            test_data_set[count][1] = whole_negative_index[i][1]
            test_data_set[count][2] = 0
            count += 1

        DTItrain, DTItest = train_data_set, test_data_set
        DTItraintest_dict['DTItrain'] = DTItrain
        DTItraintest_dict['DTItest'] = DTItest
            
    return DTItraintest_dict
