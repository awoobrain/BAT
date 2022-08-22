from ospark.nn.metrics import Metrics
from BAT.metrics.confusion_matrix import ConfusionMatrix
from typing import NoReturn, Optional
import numpy as np
import tensorflow as tf


class MacroF2(Metrics, ConfusionMatrix):
    """
    Mean all f2 scores.
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
        Calculate macroF2.
        
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
        Calculate f2 and avg f2.

        Returns:
            output: dict
        """

        collect_f1 = []
        for i in range(len(self.confusion_matrix)):
            tp = self.confusion_matrix[i][i]
            fp = np.sum(self.confusion_matrix, axis=0)[i] - tp
            fn = np.sum(self.confusion_matrix, axis=1)[i] - tp
            
            recall    = tp / (tp + fn)
            precision = tp / (tp + fp)
            
            f1 = 5 * recall * precision / (recall + 4 * precision)
            collect_f1.append(f1)
        avg_f1 = sum(collect_f1) / len(collect_f1)
        return {f"{type(self).__name__}": round(avg_f1, 5)}
