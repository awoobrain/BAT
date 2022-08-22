from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from typing import Optional
import tensorflow as tf


class BATWarmupLearningRate(LearningRateSchedule):
    """
    BAT learning rate is simliar to transformer warmup learning rate. 
    Can see on:https://arxiv.org/abs/2206.07264
    """

    def __init__(self, 
                 model_dimension: int, 
                 warmup_step: Optional[float]=None, 
                 alpha: Optional[float]=None, 
                 beta: Optional[float]=None):
        """
        Args:
            model_dimension: int
                The bigger model dimension is, the fewer learning rate is.
            warmup_step: Optional[float]
                Max learning rate happens on warmup step.
                default: 4000.
            alpha: Optional[float]
                The parameter to control the lr curve.
                default: -0.5
            beta: Optional[float]
                THe parameter to control the lr curve.
                default: 0.

        """

        self.model_dimension = tf.cast(model_dimension, dtype=tf.float32)
        self.warmup_step     = warmup_step or 4000.
        self.alpha           = alpha or -0.5
        self.beta            = beta or 0.

    def __call__(self, step: float) -> tf.Tensor:
        """
        Calculate learning rate from train step.

        Args:
            step: float
                Train step.
        
        Returns:
            learning_rate: tf.Tensor
        """
        
        arg1 = tf.pow(step, self.alpha) * tf.pow(self.warmup_step, self.beta)
        arg2 = step * tf.pow(self.warmup_step, -1.5)
        
        learning_rate = tf.math.rsqrt(self.model_dimension) * tf.math.minimum(arg1, arg2)  # before warmup step: use args2, after warmup step: use args1
        return learning_rate



