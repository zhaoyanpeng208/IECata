from yacs.config import CfgNode as CN

_C = CN()

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75

_C.DRUG.PADDING = True

_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True

# Protein light attention feature extractor
_C.PROTEIN_LA = CN()
_C.PROTEIN_LA.conv_embeddings_dim = 1024
_C.PROTEIN_LA.NUM_FILTERS = [128]
_C.PROTEIN_LA.EMBEDDING_DIM = 128
_C.PROTEIN_LA.dropout=0.25
_C.PROTEIN_LA.KERNEL_SIZE=9
_C.PROTEIN_LA.PADDING = True

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 4

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048
_C.SOLVER.loss_lamba = 0.2
#_C.SOLVER.loss_balance = 'DMW'   #CBW, CSW, DMW，LDS
_C.SOLVER.CBW_beta = 0.9
_C.SOLVER.loss_balance = ''
_C.SOLVER.scheduler = 'StepLR' # 'CosineAnnealingLR', None, 'StepLR'
_C.SOLVER.StepLR_step_size = 30
_C.SOLVER.StepLR_gamma = 0.5
_C.SOLVER.loss_function = 'ev_v1'# ev_v1
# _C.SOLVER.CosineAnnealingLR_T_max = 10

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result/train"

_C.RESULT.SAVE_MODEL = True

# protein enconde
_C.protein_encode = CN()
_C.protein_encode.input = "prottrans" #integer,prottrans
_C.protein_encode.encoder = "LA" #LA,CNN

def get_cfg_defaults():
    return _C.clone()
