from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML
__C = ConfigurationNode()

# TRAINDATASET
__C.TRAINDATASET                = ConfigurationNode()
__C.TRAINDATASET.DATADIR        = ""
__C.TRAINDATASET.FILENAMES      = []
__C.TRAINDATASET.MAX_SEQ_LENGTH = 512

# TESTDATASET
__C.TESTDATASET                 = ConfigurationNode()
__C.TESTDATASET.DATADIR         = ""
__C.TESTDATASET.FILENAMES       = []
__C.TESTDATASET.MAX_SEQ_LENGTH  = 512

# BACKBONE
__C.BACKBONE            = ConfigurationNode()
__C.BACKBONE.MODEL_NAME = "bert-base-cased"
__C.BACKBONE.EMBED_DIMS = 768 # Embedding size of each token.

# LANGUAGE_MODEL
__C.LANGUAGE_MODEL               = ConfigurationNode()
__C.LANGUAGE_MODEL.NUM_HEADS     = 4
__C.LANGUAGE_MODEL.DROPOUT_RATE  = 0.0
__C.LANGUAGE_MODEL.ENCODER_LAYER = 1 
__C.LANGUAGE_MODEL.CLASS_NUM     = 3

__C.LANGUAGE_MODEL.ATT_CONV_KERNEL_SIZE  = [1,3]
__C.LANGUAGE_MODEL.FFNN_CONV_KERNEL_SIZE = 1

# # TRAINING
__C.TRAINING              = ConfigurationNode()
__C.TRAINING.SHUFFLE_DATA = False
__C.TRAINING.SAVE_TIMES   = 10
__C.TRAINING.EPOCH        = 10
__C.TRAINING.BATCH_SIZE   = 1

__C.TRAINING.LOSS                = ""
__C.TRAINING.LOSS_GAMMA          = 0.
__C.TRAINING.LOSS_ALPHA_AMPLIFY  = 1.0 

__C.TRAINING.WEIGHT_LOAD_PATH    = ""
__C.TRAINING.WEIGHT_SAVE_PATH    = ""

__C.TRAINING.LEARNING_RATE_SELECT      = "bat_lr"
__C.TRAINING.LEARNING_RATE_WARMUP_STEP = 4000.0
__C.TRAINING.LEARNING_RATE_ALPHA       = -0.5
__C.TRAINING.LEARNING_RATE_BETA        = 0.0

__C.TRAINING.USE_AUTO_GRAPH    = True
__C.TRAINING.SAVE_INIT_WEIGHTS = True
__C.TRAINING.VALID_INTERVAL    =  10


def get_cfg_defaults() -> ConfigurationNode:
    """
    Get the ConfigurationNode instance.

    Returns:
        output: ConfigurationNode
    """

    return __C.clone()