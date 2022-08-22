import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from BAT.data_generator.conll2003_data import CONLL2003DataGenerator
from BAT.model.bat import BAT
from BAT.learningrate.bat_lr import BATWarmupLearningRate
from BAT.loss import LOSS_REGISTRY
from BAT.config import get_cfg_defaults
from BAT.trainer.bat_trainer_conll2003 import BATTrainer
from BAT.utils import CONLLDataProcessor
from transformers import AutoConfig, TFAutoModel
from transformers import AutoTokenizer
from yacs.config import CfgNode as ConfigurationNode
import json
import os
import argparse


def get_parser() -> argparse.Namespace:
    """
    Parse the parameters of outside.

    Returns:
        parameters: argparse.Namespace
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config-name',type=str, default="", help='config name')
    parameters = parser.parse_args()
    return parameters

def load_config(config_file: str) -> ConfigurationNode:
    """
    Load yaml config. 

    Args:
        config_file: str
    Returns:
        output: ConfigurationNode
    """

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def main():
    parameters = get_parser()
    cfg = load_config(config_file=f"config/{parameters.config_name}")
    print(cfg)

    tokenizer       = AutoTokenizer.from_pretrained(cfg.BACKBONE.MODEL_NAME)
    backbone_config = AutoConfig.from_pretrained(cfg.BACKBONE.MODEL_NAME, num_labels=cfg.LANGUAGE_MODEL.CLASS_NUM, output_hidden_states=True)
    backbone_model  = TFAutoModel.from_pretrained(cfg.BACKBONE.MODEL_NAME, config=backbone_config)
    backbone_model.trainable = False

    train_data_processor = CONLLDataProcessor(data_files=[cfg.TRAINDATASET.DATADIR + os.sep + filename for filename in cfg.TRAINDATASET.FILENAMES] , 
                                              tokenizer=tokenizer, 
                                              max_seq_length=cfg.TRAINDATASET.MAX_SEQ_LENGTH)
    train_data           = train_data_processor.start()
    label_to_target      = train_data_processor.label_to_target
    train_data_generator = CONLL2003DataGenerator(batch_size=cfg.TRAINING.BATCH_SIZE,
                                                  shuffle=cfg.TRAINING.SHUFFLE_DATA,
                                                  data=train_data,
                                                  label_to_target=label_to_target,
                                                  backbone_model_name_or_path=cfg.BACKBONE.MODEL_NAME,
                                                  class_num=cfg.LANGUAGE_MODEL.CLASS_NUM,
                                                  tokenizer=tokenizer,
                                                  backbone_config=backbone_config,
                                                  backbone_model=backbone_model)
    
    test_data_processor = CONLLDataProcessor(data_files=[cfg.TESTDATASET.DATADIR + os.sep + filename for filename in cfg.TESTDATASET.FILENAMES] , 
                                             tokenizer=tokenizer, 
                                             max_seq_length=cfg.TESTDATASET.MAX_SEQ_LENGTH)
    test_data           = test_data_processor.start()
    test_data_generator = CONLL2003DataGenerator(batch_size=cfg.TRAINING.BATCH_SIZE,
                                                 shuffle=False,
                                                 data=test_data,
                                                 label_to_target=label_to_target,
                                                 backbone_model_name_or_path=cfg.BACKBONE.MODEL_NAME,
                                                 class_num=cfg.LANGUAGE_MODEL.CLASS_NUM,
                                                 tokenizer=tokenizer,
                                                 backbone_config=backbone_config,
                                                 backbone_model=backbone_model)

    if cfg.TRAINING.LEARNING_RATE_SELECT=="bat_lr":
        warmup_lr = BATWarmupLearningRate(model_dimension=cfg.BACKBONE.EMBED_DIMS, 
                                          warmup_step=cfg.TRAINING.LEARNING_RATE_WARMUP_STEP,
                                          alpha=cfg.TRAINING.LEARNING_RATE_ALPHA,
                                          beta=cfg.TRAINING.LEARNING_RATE_BETA)

    optimizer = tf.keras.optimizers.Adam(warmup_lr, 
                                         beta_1=0.9, 
                                         beta_2=0.98, 
                                         epsilon=1e-9)
    training_loss = cfg.TRAINING.LOSS
    if training_loss=="ce_loss":
        loss_fn = LOSS_REGISTRY[training_loss]()
    else:
        if training_loss=="cecla_loss" and cfg.TRAINING.USE_AUTO_GRAPH:
            training_loss = training_loss+"_graph"
        elif training_loss=="cecla_loss" and not cfg.TRAINING.USE_AUTO_GRAPH:
            training_loss = training_loss+"_eager"
        loss_fn = LOSS_REGISTRY[training_loss](alpha_amplify=cfg.TRAINING.LOSS_ALPHA_AMPLIFY, gamma=cfg.TRAINING.LOSS_GAMMA, dataset=train_data_generator.dataset)

    try:
        with open(cfg.TRAINING.WEIGHT_LOAD_PATH, "r") as fp:
            weights = json.load(fp)
    except:
        weights = None
    
    if not os.path.isdir(cfg.TRAINING.WEIGHT_SAVE_PATH.split("/")[0]):
        os.mkdir(cfg.TRAINING.WEIGHT_SAVE_PATH.split("/")[0])

    bat = BAT(obj_name="BAT", 
              block_number=cfg.LANGUAGE_MODEL.ENCODER_LAYER,
              embedding_size=cfg.BACKBONE.EMBED_DIMS, 
              head_number=cfg.LANGUAGE_MODEL.NUM_HEADS, 
              scale_rate=4,
              att_conv_kernel_size=cfg.LANGUAGE_MODEL.ATT_CONV_KERNEL_SIZE,
              ffnn_conv_kernel_size=cfg.LANGUAGE_MODEL.FFNN_CONV_KERNEL_SIZE,
              dropout_rate=cfg.LANGUAGE_MODEL.DROPOUT_RATE,
              classify_num=cfg.LANGUAGE_MODEL.CLASS_NUM,
              trained_weights=weights,
              max_sequence_length=1000)#train_data_generator._dataset.page_max_length)

    trainer = BATTrainer(train_data_generator=train_data_generator,
                         evaluation_data_generator=test_data_generator,
                         model=bat,
                         use_positional_encoding=False,
                         epoch_number=cfg.TRAINING.EPOCH,
                         optimizer=optimizer,
                         loss_function=loss_fn,
                         save_delegate=None,
                         save_times=cfg.TRAINING.SAVE_TIMES,
                         save_path=cfg.TRAINING.WEIGHT_SAVE_PATH,
                         use_auto_graph=cfg.TRAINING.USE_AUTO_GRAPH,
                         save_init_weights=cfg.TRAINING.SAVE_INIT_WEIGHTS,
                         init_weights_path=None)

    print(trainer.count_parameters())
    trainer.start()


if __name__ == '__main__':
    main()