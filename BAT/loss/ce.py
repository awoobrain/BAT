from ospark.nn.loss_function import LossFunction
from . import RegisterMeta
import tensorflow as tf


class CategoricalCrossentropy(LossFunction, metaclass=RegisterMeta):
    """
    CategoricalCrossentropy loss.
    """

    register_name = "ce_loss"

    def __init__(self):
        self._loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        """
        Calculate loss value with loss function.

        Args:
            prediction: tf.Tensor
            target_data: tf.Tensor
        
        Returns:
            output: tf.Tensor
        """

        return self._loss(y_true=target_data, y_pred=prediction)/tf.cast(tf.math.count_nonzero(target_data)*target_data.shape[-1],dtype=tf.float32)
