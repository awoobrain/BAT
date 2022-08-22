from ospark.nn.component.weight import WeightOperator
from ospark.data.generator import DataGenerator
from ospark.data.save_delege import SaveDelegate
from typing import NoReturn, Optional, Callable, List, Tuple
from tensorflow.keras.optimizers import Optimizer
from ospark.nn.loss_function import LossFunction
from ospark.nn.model import Model
from ospark.trainer import *
import tensorflow as tf
import json
import time
import datetime
import seqeval.metrics
import numpy as np


class BATTrainer(Trainer):
    """
    Build BAT trainer.
    """

    def __init__(self,
                 train_data_generator: DataGenerator,
                 evaluation_data_generator: DataGenerator,
                 model: Model,
                 use_positional_encoding: bool,
                 epoch_number: int,
                 optimizer: Optimizer,
                 loss_function: LossFunction,
                 save_times: int,
                 save_path: Optional[str]=None,
                 save_delegate: Optional[SaveDelegate]=None,
                 use_auto_graph: Optional[bool]=None,
                 save_init_weights: Optional[bool]=None,
                 init_weights_path: Optional[str]=None
                 ):
        """
        Args:
            train_data_generator: DataGenerator
            evaluation_data_generator: DataGenerator
            model: Model
            use_positional_encoding: bool
            epoch_number: int
            optimizer: Optimizer
            loss_function: LossFunction
            save_times: int
            save_path: Optional[str]
                default: "weight/weights.json"
            save_delegate: Optional[SaveDelegate]
            use_auto_graph: Optional[bool]
                default: True
            save_init_weights: Optional[bool]
                default: True
            init_weights_path: Optional[str]
                default: save_path.split(".")[0] + "_init.json"
        """

        super().__init__(model=model,
                         data_generator=train_data_generator,
                         epoch_number=epoch_number,
                         optimizer=optimizer,
                         loss_function=loss_function,
                         save_delegate=save_delegate,
                         save_path=save_path or "weight/weights.json",
                         save_times=save_times,
                         use_auto_graph=True if use_auto_graph is None else use_auto_graph)
        self._use_positional_encoding     = use_positional_encoding
        self._save_init_weights           = True if save_init_weights is None else save_init_weights
        self._evaluation_data_generator   = evaluation_data_generator
        self._init_weights_path           = init_weights_path or self.save_path.split(".")[0] + "_init.json"

    def start(self) -> NoReturn:
        """
        The beginning of the training pipeline.
        """

        print("=" * 24)
        print("Training start.")
        self.training_process(epoch_number=self.epoch_number)
        print("Training end.")
        print("=" * 24)

    def training_process(self, epoch_number: int) -> NoReturn:
        """
        copyright:
        We follow the code from https://github.com/wzhouad/NLL-IE to build evaluation.
        The reason is good reproducibility, easy to maintain, and easy to demo.

        Main pipeline.

        Args:
            epoch_number: int
        """

        max_f1 = 0
        max_f1_epoch = 0
        for epoch in range(epoch_number):
            total_loss_value = 0
            training_count   = 0
            start_time       = time.time()
            print("epoch", epoch)
            for idx,batch in enumerate(self.data_generator):
                training_data, attention_mask, target_data = batch["input_embeddings"], batch["attention_mask"], batch["targets"]
                loss_value = self.training_method(training_data, attention_mask, target_data)
                total_loss_value += loss_value
                training_count   += 1
            if self._evaluation_data_generator:
                preds, keys = [], []
                for batch in self._evaluation_data_generator:
                    logits = self.model.pipeline(input_data=batch["input_embeddings"], 
                                                 encoder_padding_mask=batch["attention_mask"], 
                                                 use_positional_encoding=self._use_positional_encoding)
                    keys  += batch['no_onehot_targets'].numpy().flatten().tolist()
                    preds += np.argmax(logits.numpy(), axis=-1).flatten().tolist()
                preds, keys = list(zip(*[[pred, key] for pred, key in zip(preds, keys) if key != -1]))
                print("preds",len(preds))
                print("keys",len(keys))
                preds  = [self._evaluation_data_generator.dataset.target_to_label[pred] for pred in preds]
                keys   = [self._evaluation_data_generator.dataset.target_to_label[key] for key in keys]
                f1     = seqeval.metrics.f1_score([keys], [preds])
                max_f1 = max(max_f1, f1)
                max_f1_epoch = epoch + 1 if max_f1==f1 else max_f1_epoch
                output = {
                    "test_f1": f1,
                }
                print(output)

            print(f'Epoch {epoch + 1} '
                  f'Loss {total_loss_value / training_count:.4f} ')
            print(f'Time taken for 1 epoch: {time.time() - start_time:.2f} secs\n')
            print(f"max f1:{max_f1},出現epoch:{max_f1_epoch}")
            if max_f1_epoch == (epoch + 1):
                self.save_delegate.save(weights=self.weights_operator.weights, path=self.save_path.split(".")[0] + f"_best_f1.json") 

            if self.will_save(epoch_number=epoch) and self.save_path is not None:
                self.save_delegate.save(weights=self.weights_operator.weights, path=self.save_path.split(".")[0] + f"_epoch{epoch+1}.json") 

        if self.save_path is not None:
            self.save_delegate.save(weights=self.weights_operator.weights, path=self.save_path)
    
    def train_step(self, 
                   train_data: tf.Tensor, 
                   attention_mask: Optional[tf.Tensor], 
                   target_data: tf.Tensor) -> tf.Tensor:
        """
        Build model forward and backward.

        Args:
            train_data: tf.Tensor
            attention_mask: Optional[tf.Tensor]
            target_data: tf.Tensor
        
        Returns:
            loss_value: tf.Tensor        
        """

        with tf.GradientTape() as tape:
            prediction = self.model.pipeline(input_data=train_data, 
                                             encoder_padding_mask=attention_mask, 
                                             use_positional_encoding=self._use_positional_encoding)
            loss_value = self.loss_function(prediction=prediction, target_data=target_data)
            weights    = self.weights_operator.collect_weights()
            tape.watch(weights)
        gradients = tape.gradient(loss_value, weights)
        self.optimizer.apply_gradients(zip(gradients, weights))
        return loss_value

    def eager_mode(self,
                   train_data: tf.Tensor,
                   attention_mask: tf.Tensor,
                   target_data: tf.Tensor) -> tf.Tensor:
        """
        Args:
            train_data: tf.Tensor
            attention_mask: tf.Tensor
            target_data: tf.Tensor
        
        Returns:
            output: tf.Tensor
        """

        return self.train_step(train_data=train_data, attention_mask=attention_mask, target_data=target_data)


    @tf.function(input_signature=[
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None, 9), dtype=tf.float32)])
    def graph_mode(self,
                   train_data: tf.Tensor,
                   attention_mask: tf.Tensor,
                   target_data: tf.Tensor) -> tf.Tensor:
        """
        Args:
            train_data: tf.Tensor
                shape: (batch, seq_len, embed_dim)
            attention_mask: tf.Tensor
                shape: (batch, 1, 1, seq_len)
            target_data: tf.Tensor
                shape: (batch, seq_len, class_num)
        
        Returns:
            output: tf.Tensor
        """

        return self.train_step(train_data=train_data, attention_mask=attention_mask, target_data=target_data)

    def get_weights(self) -> dict:
        """
        Returns:
            output: dict
        """

        return self.weights_operator.weights

    def count_parameters(self) -> int:
        """
        Show the model parameters.

        Returns:
            count: int
        """

        count = 0
        weights = self.get_weights()
        for module_name, module_weights in weights.items():
            print(module_name, np.array(module_weights).shape)
            count+=np.array(module_weights).size
        return count

    def will_save(self, epoch_number: int) -> bool:
        """
        Judge save or not save.

        Args:
            epoch_number: int

        Returns:
            output: bool
        """

        if self.save_times is None:
            return False
        return (epoch_number + 1) % self.save_times == 0

    def save(self, weights: dict, path: str) -> NoReturn:
        """
        Save model weights.

        Args:
            weights: dict
            path: str
        """

        with open(path, 'w') as fp:
            json.dump(weights, fp)
