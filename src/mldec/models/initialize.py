def initialize_model(config):

    if config.get("model") == "ffnn":
        from mldec.models import ffnn
        input_dim = config.get("input_dim")
        hidden_dim = config.get("hidden_dim")
        output_dim = config.get("output_dim")
        n_layers = config.get("n_layers")
        model = ffnn.FFNN(input_dim, hidden_dim, output_dim, n_layers)
    else:
        raise ValueError("Unknown model type")
    return model