def initialize_model(config):

    if config.get("model") == "ffnn":
        from mldec.models import ffnn
        input_dim = config.get("input_dim")
        hidden_dim = config.get("hidden_dim")
        output_dim = config.get("output_dim")
        n_layers = config.get("n_layers")
        dropout = config.get("dropout", 0)
        device = config.get("device")
        model = ffnn.FFNN(input_dim, hidden_dim, output_dim, n_layers, dropout, device=device)

    elif config.get("model") == "encdec":
        from mldec.models import encdec
        input_dim = config.get("input_dim")
        output_dim = config.get("output_dim")
        d_model = config.get("d_model")
        nhead = config.get("nhead")
        num_encoder_layers = config.get("num_encoder_layers")
        num_decoder_layers = config.get("num_decoder_layers")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout", 0)
        device = config.get("device")
        model = encdec.BinarySeq2Seq(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, device=device)
    else:
        raise ValueError("Unknown model type")
    
    return model