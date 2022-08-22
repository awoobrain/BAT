from ospark.nn.loss_function import LossFunction
from ospark.data.generator import DataGenerator
from typing import Tuple, Optional, List
from . import RegisterMeta
import tensorflow as tf


class CECLA(LossFunction):
    """
    CECLA: Cross-Entropy by Contrastive Learning with Array weight
    """
    
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

        self._alpha_amplify            = alpha_amplify or 1.0
        self._gamma                    = gamma or 1.0
        self._alpha, self._beta_matrix = self.get_loss_params(dataset)
        self._beta                     = None

    def get_loss_params(self, dataset: DataGenerator) -> Tuple[List[float], List[List[float]]]:
        """
        Get parameters in loss function.

        Args:
            dataset: DataGenerator
        
        Returns:
            output: Tuple[List[float], List[List[float]]]
        """

        N         = sum(dataset.class_statistic)  # total samples
        class_num = len(dataset.class_statistic)
        alpha     = [N / N_i for N_i in dataset.class_statistic]
        alpha     = list(map(lambda data: data * self._alpha_amplify, alpha))
        beta      = [[N / ((class_num - 1) * N_i) for N_i in dataset.class_statistic] for _ in range(class_num)]
        for i in range(len(beta)):
            beta[i][i] = 0
        return alpha, beta

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        """
        Args:
            prediction: tf.Tensor
                shape: (batch, seq_len, class_num)
            target_data: tf.Tensor
                shape: (batch, seq_len, class_num)
        
        Returns:
            output: tf.Tensor
        """
        
        raise NotImplemented()


class CECLAEager(CECLA, metaclass=RegisterMeta):
    """
    Cross-Entropy by Contrastive Learning with Array weight.(only use on eager mode, and the code implemented same as paper.)
    """

    register_name = "cecla_loss_eager"

    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        """
        Loss = ai * (yt * (1-p)^gamma * log(p) + (1 - yt) * p^gamma * log(1-p))
             = yt * alpha * (1-p)^gamma * log(p) + beta * (1 - yt) * p^gamma * log(1-p)

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

        prediction = tf.reshape(prediction, shape=[-1, prediction.shape[-1]])  # (batch, seq_len, class_num)->(batch*seq_len, class_num)
        prediction = tf.transpose(prediction, perm=[1, 0])  # (batch*seq_len, class_num)->(class_num, batch*seq_len)
        
        target_data = tf.reshape(target_data, shape=[-1, target_data.shape[-1]])  # (batch, seq_len, class_num)->(batch*seq_len, class_num)
        target_data = tf.transpose(target_data, perm=[1, 0])  # (batch*seq_len, class_num)->(class_num, batch*seq_len)
 
        self._alpha = tf.reshape(self._alpha, [-1,1])
        self._beta  = tf.matmul(self._beta_matrix, target_data)  # shape = (class_num, class_num)X(class_num, batch*seq_len)=(class_num, batch*seq_len)

        positive_punishment = tf.math.multiply(target_data, tf.math.multiply(self._alpha, tf.math.multiply(tf.pow(tf.subtract(1., prediction), self._gamma), tf.math.log(tf.clip_by_value(prediction, 1e-10, 1.0)))))
        negative_punishment = tf.math.multiply(self._beta, tf.math.multiply(tf.subtract(1., target_data), tf.math.multiply(tf.pow((prediction), self._gamma), tf.math.log(tf.clip_by_value(tf.subtract(1., prediction), 1e-10, 1.0)))))
        return -tf.reduce_sum(positive_punishment + negative_punishment) / (tf.reduce_sum(target_data) * target_data.shape[-2])


class CECLAGraph(CECLA, metaclass=RegisterMeta):
    """
    Cross-Entropy by Contrastive Learning with Array weight(can use tf Graph mode)
    """

    register_name = "cecla_loss_graph"

    def __init__(self, 
                 dataset: DataGenerator, 
                 alpha_amplify: Optional[float]=None, 
                 gamma: Optional[float]=None):
        """
        Args:
            dataset: DataGenerator
            alpha_amplify: Optional[float]
            gamma: Optional[float]
        """

        super().__init__(alpha_amplify=alpha_amplify, gamma=gamma, dataset=dataset)
        self._beta_matrix = self.transpose(self._beta_matrix)

    def transpose(self, inputs:list) -> list:
        """
        Do transpose of inputs.

        Args:
            inputs: list
        
        Returns:
            output: list
        """

        return [list(i) for i in zip(*inputs)]


    def calculate(self, prediction: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        """
        Loss = yt * alpha * (1-p)^gamma * log(p) + (1 - yt) * beta * p^gamma * log(1-p)
        The formula  _beta = target_data X _beta_matrix is different from CrossEntropyByConstrastiveLearningWithArrayweightEager uesd, but the result is the same.
        
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
        self._beta  = tf.matmul(target_data, self._beta_matrix)  # shape = (batch,N,3)X(3,3)=(batch,N,3)

        positive_punishment = tf.math.multiply(target_data, tf.math.multiply(self._alpha, tf.math.multiply(tf.pow(tf.subtract(1., prediction), self._gamma), tf.math.log(tf.clip_by_value(prediction, 1e-10, 1.0)))))
        negative_punishment = tf.math.multiply(tf.subtract(1., target_data), tf.math.multiply(self._beta, tf.math.multiply(tf.pow((prediction), self._gamma), tf.math.log(tf.clip_by_value(tf.subtract(1., prediction), 1e-10, 1.0)))))
        return -tf.reduce_sum(positive_punishment + negative_punishment) / (tf.reduce_sum(target_data) * target_data.shape[-1])