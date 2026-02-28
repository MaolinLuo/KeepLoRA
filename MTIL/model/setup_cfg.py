from yacs.config import CfgNode as CN


def reset_cfg(cfg, args):
    cfg.config_path = args.config_path
    cfg.gpu_id = args.gpu_id
    

def extend_cfg(cfg):
    """
    Add config variables.
    """
    cfg.dataset_root = ""
    cfg.model_backbone_name = ""
    cfg.prec = "fp16"
    cfg.input_size = (-1, -1)
    cfg.prompt_template = ""
    cfg.add_info = ""
    cfg.dataset = ""
    cfg.seed = -1
    cfg.eval_only = False

    cfg.zero_shot = False
    cfg.tasks = []
    cfg.v_svd_param_names = None
    cfg.v_svd_alphas = None
    cfg.t_svd_param_names = None
    cfg.t_svd_alphas = None

    cfg.optim = CN()
    cfg.optim.batch_size = 64
    cfg.optim.name = "SGD"
    cfg.optim.lr = None
    cfg.optim.epoch_list = []
    cfg.optim.weight_decay = None
    cfg.optim.lr_scheduler = ""

    cfg.v_full_tuning = False  # full fine-tuning
    cfg.v_keeplora = None
    cfg.v_partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
    cfg.v_adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.
    cfg.v_prin_subspace_threshold = 0.99
    cfg.v_prin_subspace_threshold2 = 0.8

    cfg.t_full_tuning = False  # full fine-tuning
    cfg.t_keeplora = None
    cfg.t_partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
    cfg.t_adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.
    cfg.t_prin_subspace_threshold = 0.99
    cfg.t_prin_subspace_threshold2 = 0.8

    cfg.classifier = None  # 'LinearClassifier' or 'CosineClassifier' or None


def setup_cfg(args):
    cfg = CN()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config_path)

    # From input arguments
    reset_cfg(cfg, args)

    # From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg


def print_args(args, cfg):
    print("** Arguments **")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("** Config **")
    print(cfg)