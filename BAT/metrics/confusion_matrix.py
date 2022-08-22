import numpy as np
import tensorflow as tf


class ConfusionMatrix:
    """
    A specific table layout that allows visualization of the performance of an algorithm.
    """

    def process(self, prediction: tf.Tensor, target: tf.Tensor) -> dict:
        """
        Run confusion matrix

        Args:
            prediction: tf.Tensor
                shape: (seq_len, class_num)
            target: tf.Tensor
                shape: (seq_len, class_num)

        Returns:
            output: dict
        """

        return {f"{type(self).__name__}": self.get_confusion_matrix(prediction=prediction, target=target)}

    def get_confusion_matrix(self, prediction: tf.Tensor, target: tf.Tensor) -> list:
        """
        Get confusion matrix

        Args:
            prediction: tf.Tensor
                shape: (seq_len, class_num)
            target: tf.Tensor
                shape: (seq_len, class_num)

        Returns:
            output: list
        """

        prediction_to_index = np.argmax(prediction, axis=-1).tolist()
        targets_to_index    = np.argmax(target, axis=-1).tolist()
        con_mat             = tf.math.confusion_matrix(labels=targets_to_index, predictions=prediction_to_index)
        return con_mat.numpy().tolist()
