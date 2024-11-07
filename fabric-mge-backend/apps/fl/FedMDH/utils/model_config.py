CONFIGS_ = {
    # input_channel, n_class, hidden_dim, latent_dim

    'material': ([6, 16, 'F'], 1, 1, 784, 32)
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    
    'material': (128, 64, 1, 64),
    
}



RUNCONFIGS = {
    'material':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            #'unique_labels': 10,    # available labels 回归问题中没有这一项
            'generative_alpha': 0.0001, # used to regulate user training
            'generative_beta': 0.0001, # used to regulate user training
            'weight_decay': 1e-2
        },

}

