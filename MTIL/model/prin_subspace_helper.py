from model.customClip import CustomCLIP


def init_prin_subspace_helper(cfg):
    prin_subspace_name_list = []
    threshold_dict = {}
    threshold_dict2 = {}

    if cfg.v_keeplora is not None:
        if 'q' in cfg.v_keeplora:
            prin_subspace_name_list.append('vit_q')
            threshold_dict['vit_q'] = cfg.v_prin_subspace_threshold
            threshold_dict2['vit_q'] = cfg.v_prin_subspace_threshold2
        if 'k' in cfg.v_keeplora:
            prin_subspace_name_list.append('vit_k')
            threshold_dict['vit_k'] = cfg.v_prin_subspace_threshold
            threshold_dict2['vit_k'] = cfg.v_prin_subspace_threshold2
        if 'v' in cfg.v_keeplora:
            prin_subspace_name_list.append('vit_v')
            threshold_dict['vit_v'] = cfg.v_prin_subspace_threshold
            threshold_dict2['vit_v'] = cfg.v_prin_subspace_threshold2
        if 'o' in cfg.v_keeplora:
            prin_subspace_name_list.append('vit_o')
            threshold_dict['vit_o'] = cfg.v_prin_subspace_threshold
            threshold_dict2['vit_o'] = cfg.v_prin_subspace_threshold2
    if cfg.t_keeplora is not None:
        if 'q' in cfg.t_keeplora:
            prin_subspace_name_list.append('text_q')
            threshold_dict['text_q'] = cfg.t_prin_subspace_threshold
            threshold_dict2['text_q'] = cfg.t_prin_subspace_threshold2
        if 'k' in cfg.t_keeplora:
            prin_subspace_name_list.append('text_k')
            threshold_dict['text_k'] = cfg.t_prin_subspace_threshold
            threshold_dict2['text_k'] = cfg.t_prin_subspace_threshold2
        if 'v' in cfg.t_keeplora:
            prin_subspace_name_list.append('text_v')
            threshold_dict['text_v'] = cfg.t_prin_subspace_threshold
            threshold_dict2['text_v'] = cfg.t_prin_subspace_threshold2
        if 'o' in cfg.t_keeplora:
            prin_subspace_name_list.append('text_o')
            threshold_dict['text_o'] = cfg.t_prin_subspace_threshold
            threshold_dict2['text_o'] = cfg.t_prin_subspace_threshold2

    return prin_subspace_name_list, threshold_dict, threshold_dict2


def get_accumulated_weight_matrix_list(model: CustomCLIP, prin_subspace_name_list: list):
    feature_matrix = dict()

    emb_dim = model.image_encoder.positional_embedding.shape[1]
    if 'vit_q' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.image_encoder.blocks:
            matrix_list.append(module_dict.attn.in_proj_weight[:emb_dim, :])
        feature_matrix['vit_q'] = matrix_list
    if 'vit_k' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.image_encoder.blocks:
            matrix_list.append(module_dict.attn.in_proj_weight[emb_dim:2*emb_dim, :])
        feature_matrix['vit_k'] = matrix_list
    if 'vit_v' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.image_encoder.blocks:
            matrix_list.append(module_dict.attn.in_proj_weight[2*emb_dim:, :])
        feature_matrix['vit_v'] = matrix_list
    if 'vit_o' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.image_encoder.blocks:
            matrix_list.append(module_dict.attn.out_proj.weight)
        feature_matrix['vit_o'] = matrix_list
    
    emb_dim = model.text_encoder.positional_embedding.shape[1]
    if 'text_q' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.text_encoder.blocks:
            matrix_list.append(module_dict.attn.in_proj_weight[:emb_dim, :])
        feature_matrix['text_q'] = matrix_list
    if 'text_k' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.text_encoder.blocks:
            matrix_list.append(module_dict.attn.in_proj_weight[emb_dim:2*emb_dim, :])
        feature_matrix['text_k'] = matrix_list
    if 'text_v' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.text_encoder.blocks:
            matrix_list.append(module_dict.attn.in_proj_weight[2*emb_dim:, :])
        feature_matrix['text_v'] = matrix_list
    if 'text_o' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.text_encoder.blocks:
            matrix_list.append(module_dict.attn.out_proj.weight)
        feature_matrix['text_o'] = matrix_list
    return feature_matrix


def get_accumulated_feature_matrix_list(model: CustomCLIP, prin_subspace_name_list: list):
    feature_matrix = dict()

    if 'vit_q' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.v_tuner.keeplora_list:
            if module_dict is not None:
                matrix_list.append(module_dict['q'].accumulated_feature_matrix)
        feature_matrix['vit_q'] = matrix_list
    if 'vit_k' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.v_tuner.keeplora_list:
            if module_dict is not None:
                matrix_list.append(module_dict['k'].accumulated_feature_matrix)
        feature_matrix['vit_k'] = matrix_list
    if 'vit_v' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.v_tuner.keeplora_list:
            if module_dict is not None:
                matrix_list.append(module_dict['v'].accumulated_feature_matrix)
        feature_matrix['vit_v'] = matrix_list
    if 'vit_o' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.v_tuner.keeplora_list:
            if module_dict is not None:
                matrix_list.append(module_dict['o'].accumulated_feature_matrix)
        feature_matrix['vit_o'] = matrix_list
    if 'text_q' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.t_tuner.keeplora_list:
            if module_dict is not None:
                matrix_list.append(module_dict['q'].accumulated_feature_matrix)
        feature_matrix['text_q'] = matrix_list
    if 'text_k' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.t_tuner.keeplora_list:
            if module_dict is not None:
                matrix_list.append(module_dict['k'].accumulated_feature_matrix)
        feature_matrix['text_k'] = matrix_list
    if 'text_v' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.t_tuner.keeplora_list:
            if module_dict is not None:
                matrix_list.append(module_dict['v'].accumulated_feature_matrix)
        feature_matrix['text_v'] = matrix_list
    if 'text_o' in prin_subspace_name_list:
        matrix_list = []
        for module_dict in model.t_tuner.keeplora_list:
            if module_dict is not None:
                matrix_list.append(module_dict['o'].accumulated_feature_matrix)
        feature_matrix['text_o'] = matrix_list
    return feature_matrix

