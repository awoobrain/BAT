from ospark.nn.loss_function import LossFunction
from ospark.data.generator import DataGenerator
from typing import Tuple, Optional, List
from . import RegisterMeta
import tensorflow as tf


class PunishBadPrediction(LossFunction, metaclass=RegisterMeta):
    """
    Punish Bad Prediction.
    """

    register_name = "pbp_loss"

    def __init__(self, 
                 dataset: DataGenerator, 
                 alpha_amplify: Optional[float]=None, 
                 gamma: Optional[float]=None):
        """
        Args:
            dataset: DataGenerator
            alpha_amplify: Optional[float]
                default: 1.0
            gamma: Optional[float]
                default: 1.0
        """

        self._alpha_amplify     = alpha_amplify or 1.0
        self._gamma             = gamma or 1.0
        self._alpha, self._beta = self.get_loss_params(dataset)

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        """
        loss = yt * alpha * (1-p)^gamma * log(p) + beta * ((1 - yt) ⊙ y_hat) * p^gamma * log(1-p)
        
        Args:
            prediction: tf.Tensor
                shape: (batch, seq_len, class_num)
            target_data: tf.Tensor
                shape: (batch, seq_len, class_num)
        
        Returns:
            output: tf.Tensor
        """

        prediction  = tf.cast(prediction, dtype=tf.float32)
        target_data = tf.cast(target_data, dtype=tf.float32)

        positive_punishment = tf.math.multiply(target_data, tf.math.multiply(self._alpha, tf.math.multiply(tf.pow(tf.subtract(1., prediction), self._gamma), tf.math.log(tf.clip_by_value(prediction, 1e-10, 1.0)))))
        
        predict_class = tf.math.argmax(prediction, axis=tf.rank(target_data) - 1)
        predict_class = tf.one_hot(predict_class, prediction.shape[-1], on_value=None, off_value=None, axis=None, dtype=None, name=None)
        
        negative_punishment = tf.math.multiply(tf.math.multiply(tf.subtract(1., target_data), predict_class), tf.math.multiply(tf.pow((prediction), self._gamma), tf.math.log(tf.clip_by_value(tf.subtract(1., prediction), 1e-10, 1.0))))
        negative_punishment = tf.math.multiply(self._beta, negative_punishment)
        return (-tf.reduce_sum(positive_punishment + negative_punishment) / tf.cast(tf.math.count_nonzero(target_data) * target_data.shape[-1], dtype=tf.float32))

    def get_loss_params(self, dataset: DataGenerator) -> Tuple[List[float], List[float]]:
        """
        Get parameters in loss function.

        Args:
            dataset: DataGenerator
        
        Returns:
            output: Tuple[List[float], List[float]]
        """

        N     = sum(dataset.class_statistic) # total samples
        alpha = [N / N_i for N_i in dataset.class_statistic]
        alpha = list(map(lambda data: data * self._alpha_amplify, alpha))
        beta  = [N / (N - N_i) for N_i in dataset.class_statistic]
        return alpha, beta