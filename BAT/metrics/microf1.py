from ospark.nn.metrics import Metrics
from BAT.metrics.confusion_matrix import ConfusionMatrix
from typing import NoReturn, Optional
import numpy as np
import tensorflow as tf

class MicroF1(Metrics, ConfusionMatrix):
    """
    Calculate metrics globally by counting the total true positives, false negatives and false positives.
    """

    def __init__(self, confusion_matrix: Optional[list]=None):
        """
        Args:
            confusion_matrix: Optional[list]
        """

        self._confusion_matrix = confusion_matrix

    @property
    def confusion_matrix(self) -> list:
        return self._confusion_matrix

    def reset_confusion_matrix(self) -> NoReturn:
        """
        Reset confusion matrix.
        """

        self._confusion_matrix = None

    def process(self, prediction: tf.Tensor, target: tf.Tensor) -> dict:
        """
        Calculate microF1.

        Args:
            prediction: tf.Tensor
                shape: (seq_len, class_num)
            target: tf.Tensor
                shape: (seq_len, class_num)

        Returns:
            output: dict
        """

        if not self.confusion_matrix:
            self._confusion_matrix = self.get_confusion_matrix(prediction=prediction, target=target)
        output = self.calculate_start()
        self.reset_confusion_matrix()
        return output

    def calculate_start(self) -> dict:
        """
        Calculate microf1.

        Returns:
            output: dict
        """

        TP = np.sum(np.diagonal(self.confusion_matrix))
        FP = np.sum(self.confusion_matrix) - TP
        FN = np.sum(self.confusion_matrix) - TP
        
        recall    = TP / (TP + FN)
        precision = TP / (TP + FP)
        
        get_f1_score = 2 * (recall * precision) / (recall + precision) 
        return {f"{type(self).__name__}": round(get_f1_score, 5)}
