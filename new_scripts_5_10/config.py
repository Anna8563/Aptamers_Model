def get_config():
    # Default configuration
    config = {
        'experiment_name': 'New_Transformer_9_10',
        'model_basename': 'New_Transformer_9_10',
        'batch_size': 32,
        'num_epochs': 100,
        'save_every': 1,
        'lr': 1e-4,
        'seq_len': 80,
        'num_heads': 8,
        'num_layers': 2,
        'd_model': 512,
        'd_ff': 256,
        'input_dim':2304,
        'model_folder': 'checkpoints',
        'kmer': 1,
        'comments': 'New_Transformer 9_10_25'
    }
    return config