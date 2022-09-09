def prepare_dataloader(matrix, n_col, seq_len=36, pred_len=12, BATCH_SIZE=32, device='cpu'):
    seg = matrix.columns.values
    time = matrix.index.values
    n_seg = len(seg)
    n_time = len(time)
    
    speedMatrix = matrix.to_numpy()
    
    data_set = []
    label_set = []

    for i in range(n_time - seq_len - pred_len):
        data = speedMatrix[i : i + seq_len]
        
        label_data = speedMatrix[i + seq_len: i + seq_len + pred_len, :n_col]
        
        if np.isnan(np.sum(data[:n_col])).any() | np.isnan(np.sum(label_data)):
            pass
        else:

            data_set.append(data)
            label_set.append(label_data)
            
    data = np.array(data_set)
    label = np.array(label_set)

    train_ind = int(len(data)* 0.8)
    valid_ind = int(len(data) * 0.9)
    test_ind = int(len(data) * 1.0)

    X_train = data[: train_ind]
    X_valid = data[train_ind : valid_ind]
    X_test = data[valid_ind : test_ind]
    Y_train = label[: train_ind]
    Y_valid = label[train_ind : valid_ind]
    Y_test = label[valid_ind : test_ind]

    X_train = torch.FloatTensor(X_train).to(device)
    print(X_train.size())
    X_valid = torch.FloatTensor(X_valid).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    Y_train = torch.FloatTensor(Y_train).to(device)
    Y_valid = torch.FloatTensor(Y_valid).to(device)
    Y_test = torch.FloatTensor(Y_test).to(device)

    train_dataset = utils.TensorDataset(X_train, Y_train)
    valid_dataset = utils.TensorDataset(X_valid, Y_valid)
    test_dataset = utils.TensorDataset(X_test, Y_test)

    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = False)

    return train_dataloader, valid_dataloader, test_dataloader
    
data_matrix = load_data("./tps_df.pkl", "2020-01-01 00:00:00.000", "2020-05-31 23:45:00.000", freq="15min")
input_dim = data_matrix.shape[-1]
train_dataloader, valid_dataloader, _ = prepare_dataloader(data_matrix, input_dim, BATCH_SIZE=BATCH_SIZE, seq_len=INPUT_LEN, pred_len=PRED_LEN, device="cpu")
