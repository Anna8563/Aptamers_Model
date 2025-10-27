def get_config():
    # Default configuration
    config = {
        'experiment_name': 'Test_Transformer_with_mirna',
        'model_basename': 'Test_Transformer_with_mirna',
        'batch_size': 32,
        'num_epochs': 100,
        'save_every': 50,
        'dropout': 0.1,
        'lr': 1e-4,
        'seq_len': 182,  #max_len + 2 for SOS and EOS tokens
        'num_heads': 8,
        'num_layers': 2,
        'd_model': 1280,
        'd_ff': 256,
        'input_dim':2304,
        'model_folder': 'checkpoints',
        'preload': True,
        'kmer': 1,
        'comments': 'Test_Transformer_with_mirna',
        'patience': 10, #EarlyStopping
        'delta_for_early_stop': 0.01 #EarlyStopping
    }
    return config