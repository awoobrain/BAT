import tensorflow as tf
import ospark
from ospark.nn.component.basic_module import ModelObject
from ospark import weight_initializer
from typing import NoReturn, List, Optional, Callable, Union


class Normalization(ModelObject):
    """
    Applies Normalization on a mini-batch of input features.
    """

    def __init__(self,
                 obj_name: str,
                 layer_dimension: Union[int, list],
                 epsilon: Optional[float]=None,
                 use_bias: Optional[bool]=None,
                 use_scale: Optional[bool]=None):
        """
        Args:
            obj_name: str,
            layer_dimension: Union[int, list]
            epsilon: Optional[float]
                default: 0.0001
            use_bias: Optional[bool]
                default: True
            use_scale: Optional[bool]
                default: True
        """

        super().__init__(obj_name=obj_name)
        self._gamma           = None
        self._beta            = None
        self._epsilon         = epsilon or 0.0001
        self._use_bias        = True if use_bias is None else use_bias
        self._use_scale       = True if use_scale is None else use_scale
        self._layer_dimension = layer_dimension

    @property
    def gamma(self) -> tf.Tensor:
        return self._gamma

    @property
    def beta(self) -> tf.Tensor:
        return self._beta

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def use_scale(self) -> bool:
        return self._use_scale

    @property
    def epsilon(self) -> tf.Tensor:
        return tf.constant(self._epsilon)

    @property
    def layer_dimension(self) -> tf.Tensor:
        return self._layer_dimension

    def in_creating(self) -> NoReturn:
        """
        Build weights in normalization.
        """

        if self.use_scale:
            self._gamma = weight_initializer.ones(obj_name="gamma", shape=self.layer_dimension)
        if self.use_bias:
            self._beta = weight_initializer.zeros(obj_name="beta", shape=self.layer_dimension)

    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(Normalization, cls.__name__, cls)

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Run Normalization.
        
        Args:
            input_data: tf.Tensor

        Returns:
            output: tf.Tensor

        Raises:
            NotImplementedError:
                An error occurs when function is not implemented.

        """

        raise NotImplementedError()

    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Calculate the normalization result.

        Args:
            input_data: tf.Tensor
        
        Returns:
            output: tf.Tensor
        """

        return self.calculate(input_data)


class LayerNormalization(Normalization):
    """
    Applies Layer Normalization on a mini-batch of input features.
    """

    def __init__(self,
                 layer_dimension: Union[int, list],
                 obj_name: Optional[str]=None,
                 epsilon: Optional[float]=None,
                 use_bias: Optional[bool]=None,
                 use_scale: Optional[bool]=None):
        """
        Args:
            layer_dimension: Union[int, list]
            obj_name: Optional[str]
                default: "layer_norm"
            epsilon: Optional[float]
                default: 0.0001
            use_bias: Optional[bool]
                default: True
            use_scale: Optional[bool]
                default: True
        """

        super().__init__(obj_name=obj_name or "layer_norm",
                         layer_dimension=layer_dimension,
                         epsilon=epsilon or 0.0001,
                         use_bias=True if use_bias is None else use_bias,
                         use_scale=True if use_scale is None else use_bias)

    def calculate(self, input_data: tf.Tensor, axis: Optional[int]=None) -> tf.Tensor:
        """
        Run Layer Normalization.
        
        Args:
            input_data: tf.Tensor
            axis: Optional[int]
                default: -1

        Returns:
            output: tf.Tensor
        """

        mean, variance = tf.nn.moments(input_data, axes=[axis or -1], keepdims=True)
        normalization_outputs = (input_data - mean) / (tf.sqrt(variance) + self.epsilon)
        if self.use_scale:
            normalization_outputs *= self._gamma
        if self.use_bias:
            normalization_outputs += self._beta
        return normalization_outputs