from ospark.data.generator import DataGenerator
from typing import NoReturn, List, Optional, Set, Tuple, Dict
from functools import reduce
from collections import Counter
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import AutoConfig, PretrainedConfig
from transformers import TFAutoModel, TFPreTrainedModel
import tensorflow as tf 
import math
import random


class CONLL2003DataGenerator(DataGenerator):
    """
    Build conll2003 data generator.
    """

    class Dataset:
        """
        Include all training data and labels.
        """

        def __init__(self, class_num: int, label_to_target: Dict[str, int]):
            self._data_packager       = []
            self._class_statistic     = [0] * class_num
            self._class_num           = class_num
            self._label_to_target     = label_to_target
            self._page_max_length     = 0
        """
        Args:
            class_num: int
                The number of the class.
            label_to_target: Dict[str, int]
                A dict mapping label to target.
        """

        @property
        def class_statistic(self) -> list:
            return self._class_statistic

        @property
        def class_num(self) -> list:
            return self._class_num

        @property
        def label_to_target(self) -> Dict[str, int]:
            return self._label_to_target

        @property
        def target_to_label(self) -> Dict[int, str]:
            return {target: label for label, target in self._label_to_target.items()}

        @property
        def page_max_length(self) -> list:
            return self._page_max_length


    def __init__(self, 
                 batch_size: int,
                 data: List[dict],
                 label_to_target: Dict[str, int],
                 backbone_model_name_or_path: str,
                 class_num: int,
                 shuffle: Optional[bool]=None,
                 initial_step: Optional[int]=None,
                 tokenizer: Optional[PreTrainedTokenizerFast]=None,
                 backbone_config: Optional[PretrainedConfig]=None,
                 backbone_model: Optional[TFPreTrainedModel]=None,
                 ):
        """
        Args:
            batch_size: int
            data: List[dict]
            label_to_target: Dict[str, int]
                Label mapping to the class index.
            backbone_model_name_or_path: str
                Select the suitable transformer-based pretrained model to be backbone.
            class_num: int
            shuffle: Optional[bool]
                default: True
            initial_step: Optional[int]
                default: 0
            tokenizer: Optional[PreTrainedTokenizerFast]
                default: AutoTokenizer.from_pretrained(model_name)
            backbone_config: Optional[PretrainedConfig]
                default: AutoConfig.from_pretrained(model_name, num_labels, output_hidden_states)
            backbone_model: Optional[TFPreTrainedModel]
                default: TFAutoModel.from_pretrained(model_name, config)
        """

        self._batch_size      = batch_size
        self._step            = initial_step or 0
        self._dataset         = self.Dataset(class_num=class_num, label_to_target=label_to_target)
        self._tokenizer       = tokenizer or AutoTokenizer.from_pretrained(backbone_model_name_or_path)
        self._backbone_config = backbone_config or AutoConfig.from_pretrained(backbone_model_name_or_path, num_labels=class_num, output_hidden_states=True)
        self._backbone_model  = backbone_model or TFAutoModel.from_pretrained(backbone_model_name_or_path, config=self._backbone_config)
        self._shuffle         = True if shuffle is None else shuffle
        
        self.setting_dataset(data=data)
        self._max_step       = math.ceil(len(self._dataset._data_packager) / self._batch_size)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def step(self) -> int:
        return self._step

    @property
    def max_step(self) -> int:
        return self._max_step

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self.step < self.max_step:
            dataset = self._get_data()
            self._step += 1
            return dataset
        if self._shuffle:
            self.shuffle()
        self.reset()
        raise StopIteration()

    def shuffle(self) -> NoReturn:
        """
        Shuffle the list of data.
        """

        random.shuffle(self._dataset._data_packager)

    def reset(self) -> NoReturn:
        """
        Reset the step.
        """

        self._step = 0

    def _get_data(self) -> dict:
        """
        Use collate_fn and step to generate corresponding mini-batch data.

        Returns:
            output: dict
        """

        start = self.step * self.batch_size
        end   = start + self.batch_size
        return self.collate_fn(self._dataset._data_packager[start: end])

    @classmethod
    def wrapped_tf_datasets(cls, 
                            batch_size: int,
                            data: List[dict],
                            label_to_target: Dict[str, int],
                            backbone_model_name_or_path: str,
                            class_num: int,
                            shuffle: Optional[bool]=None,
                            initial_step: Optional[int]=None,
                            tokenizer: Optional[PreTrainedTokenizerFast]=None,
                            backbone_config: Optional[PretrainedConfig]=None,
                            backbone_model: Optional[TFPreTrainedModel]=None,
                            ) -> tf.data.Dataset:
        """
        Wrap data to be tf dataset.

        Args:
            batch_size: int
            data: List[dict]
            label_to_target: Dict[str, int]
            backbone_model_name_or_path: str
            class_num: int
            shuffle: Optional[bool]
            initial_step: Optional[int]
                default: 0
            tokenizer: Optional[PreTrainedTokenizerFast]
                default: AutoTokenizer.from_pretrained(model_name)
            backbone_config: Optional[PretrainedConfig]
                default: AutoConfig.from_pretrained(model_name, num_labels, output_hidden_states)
            backbone_model: Optional[TFPreTrainedModel]
                default: TFAutoModel.from_pretrained(model_name, config)
        Returns:
            output: tf.data.Dataset
        """

        generator = cls(batch_size=batch_size,
                        data=data,
                        label_to_target=label_to_target,
                        backbone_model_name_or_path=backbone_model_name_or_path,
                        class_num=class_num,
                        shuffle=shuffle,
                        initial_step=initial_step,
                        tokenizer=tokenizer,
                        backbone_config=backbone_config,
                        backbone_model=backbone_model)

        def generable() -> CONLL2003DataGenerator:
            """
            Build callable generator.

            Returns:
                generator: CONLL2003DataGenerator
            """

            return generator

        generator = tf.data.Dataset.from_generator(generable,
                                                   output_types={"input_embeddings" : tf.float32,
                                                                 "attention_mask"   : tf.float32,
                                                                 "targets"          : tf.float32,
                                                                 "no_onehot_targets": tf.int32
                                                                }
                                                   )

        return generator

    def collate_fn(self, batch_data: List[dict]) -> dict:
        """
        copyright:
        We follow the code from https://github.com/wzhouad/NLL-IE to build collate_fn.
        The reason is good reproducibility, easy to maintain, and easy to demo.

        Collate lists of samples into batches.

        Args:
            batch_data: List[dict]
                Include so many dict in it. For example:
                [{"token_ids":[101,11,12,16,102] ,"token_targets":[-1,8,2,3,-1]}, {"token_ids":[101,16,102] ,"token_targets":[-1,3,-1]},...]
        
        Returns:
            output: dict
                For example:
                {
                    "input_embeddings": shape:[batch_size, seq_len, embed_size]
                    "attention_mask": shape:[batch_size, 1, 1, seq_len]
                    "targets": shape:[batch_size, seq_len, class_num]
                    "no_onehot_targets": [batch_size, seq_len]
                }
        """

        sample_lens       = [len(sample["token_ids"]) for sample in batch_data]
        batch_max_seq_len = max(sample_lens)
        attention_mask    = [[1.0] * sample_len + [0.0] * (batch_max_seq_len - sample_len) for sample_len in sample_lens]
        token_ids         = [sample["token_ids"] + [self._tokenizer.pad_token_id] * (batch_max_seq_len - len(sample["token_ids"])) for sample in batch_data]
        targets           = [sample["token_targets"] + [-1] * (batch_max_seq_len - len(sample["token_targets"])) for sample in batch_data]

        #transfer list into tensor
        token_ids      = tf.constant(token_ids, dtype=tf.int32)
        attention_mask = tf.constant(attention_mask, dtype=tf.float32)
        targets        = tf.constant(targets, dtype=tf.int32)

        all_layers_state = self._backbone_model(token_ids, attention_mask=attention_mask)[2]
        all_layers_state = [layer_state for layer_state in all_layers_state]
        all_layers_state = tf.stack(all_layers_state, axis=0)
        
        mean_all_layers_state = tf.reduce_mean(all_layers_state, axis=0)
        
        attention_mask = 1.0 - attention_mask
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        output = {
            "input_embeddings" : mean_all_layers_state,
            "attention_mask"   : attention_mask,
            "targets"          : tf.one_hot(indices=targets, depth=self._dataset.class_num),
            "no_onehot_targets": targets
        }
        return output

    def setting_dataset(self, data: List[dict]) -> NoReturn:
        """
        Set dataset.

        Args:
            data: List[dict]
                For example:[{"token_ids":[101,55,102],"token_targets":[-1,8,-1]},...]
        """

        token_targets = list(reduce(lambda x, y: x + y["token_targets"], data, []))
        count_targets = dict(Counter(token_targets))
        for i in range(self._dataset.class_num):
            self._dataset._class_statistic[i] = count_targets[i]
        page_max_length = 0
        for datum in data:
            page_max_length = max(page_max_length, len(datum["token_targets"]))
        self._dataset._page_max_length = page_max_length
        self._dataset._data_packager   = data
        
    def show_class_statistic(self) -> NoReturn:
        """
        Show class distributed.
        """

        print(self._dataset._class_statistic)