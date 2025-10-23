def get_config():
    # Default configuration
    config = {
        'experiment_name': 'Test_Transformer_with_mirna',
        'model_basename': 'Test_Transformer_with_mirna',
        'batch_size': 32,
        'num_epochs': 100,
        'save_every': 200000,
        'dropout': 0.1,
        'lr': 1e-4,
        'seq_len': 180,
        'num_heads': 8,
        'num_layers': 2,
        'd_model': 1280,
        'd_ff': 256,
        'input_dim':2304,
        'model_folder': 'checkpoints',
        'preload': True,
        'kmer': 1,
        'comments': 'Test_Transformer_with_mirna 21_10_25',
        'patience': 10, #EarlyStopping
        'delta_for_early_stop': 0.01 #EarlyStopping
    }
    return config