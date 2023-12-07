from src.networks.decoders import Decoders

def get_model(cfg):
    in_dim = cfg['encoding']['n_levels'] * cfg['encoding']['feature_dim']  # feature dimensions
    truncation = cfg['model']['truncation']
    learnable_beta = cfg['rendering']['learnable_beta']
    use_tcnn = cfg['encoding']['tcnn']
    device = cfg['device']
    decoder = Decoders(device=device, in_dim=in_dim, truncation=truncation, learnable_beta=learnable_beta, use_tcnn=use_tcnn)

    return decoder
