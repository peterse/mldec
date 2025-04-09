def initialize_model(config):

    input_dim = config.get("input_dim")
    output_dim = config.get("output_dim")
    dropout = config.get("dropout", 0)
    device = config.get("device")

    if config.get("model") == "ffnn":
        from mldec.models import ffnn
        hidden_dim = config.get("hidden_dim")
        n_layers = config.get("n_layers")
        model = ffnn.FFNN(input_dim, hidden_dim, output_dim, n_layers, dropout, device=device)

    elif config.get("model") == "cnn":
        from mldec.models import cnn
        conv_channels = config.get("conv_channels") # number of convolution channels per layer
        n_layers = config.get("n_layers")
        kernel_size = config.get("kernel_size")
        model = cnn.CNN(input_dim, conv_channels, output_dim, n_layers, kernel_size, dropout, device=device)
    
    elif config.get("model") == "transformer":
        from mldec.models import encdec
        d_model = config.get("d_model") # d_model, or 'width' = emb_dimension=Q,K,V dimensions; divided by nhead for multihead attention
        nhead = config.get("nhead")
        num_encoder_layers = config.get("num_encoder_layers")
        num_decoder_layers = config.get("num_decoder_layers")
        dim_feedforward = config.get("dim_feedforward")
        sos, eos = config.get("sos_eos")
        model = encdec.BinarySeq2Seq(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, device=device, sos_token=sos)
    
    elif config.get("model") == "gnn":
        from mldec.models import gnn
        gcn_depth = config.get("gcn_depth")
        gcn_min = config.get("gcn_min")
        mlp_depth = config.get("mlp_depth")
        mlp_max = config.get("mlp_max")
        model = gnn.RepGNN(input_dim, output_dim, gcn_depth, gcn_min, mlp_depth, mlp_max)
    else:
        raise ValueError("Unknown model type")
    
    return model


def count_parameters(model):
	tot = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return tot, trainable
