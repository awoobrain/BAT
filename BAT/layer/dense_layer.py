from ospark.nn.layers import Layer
from ospark import weight_initializer
from typing import List, NoReturn, Optional, Callable, Tuple, Union
from BAT.component.activation import Activation, PassActivation
from functools import reduce
import tensorflow as tf
import ospark


class DenseLayer(Layer):
    """
    Transfer the embedding space into the other embedding space.
    """

    def __init__(self,
                 obj_name:str,
                 input_dimension: int,
                 output_dimension: int,
                 activation: Optional[str]=None,
                 use_bias: Optional[bool]=None):
        """
        Args:
            obj_name: str
            input_dimension: int
            output_dimension: int
            activation: Optional[str]
                default: PassActivation()
            use_bias: Optional[bool]
                default: True
        """

        super().__init__(obj_name=obj_name)
        self._input_dimension  = input_dimension
        self._output_dimension = output_dimension
        self._layers_name      = []
        self._activation       = PassActivation() if activation is None else getattr(Activation, activation)()
        self._use_bias         = True if use_bias is None else use_bias
        self._forward          = self.bias_forward if use_bias else self.no_bias_forward

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def output_dimension(self) -> List[int]:
        return self._output_dimension

    @property
    def layers_name(self) -> list:
        return self._layers_name

    @property
    def activation(self) -> Activation:
        return self._activation

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def forward(self) -> Callable[[tf.Tensor, Union[List[tf.Tensor], tf.Tensor]], tf.Tensor]:
        return self._forward

    def in_creating(self) -> NoReturn:
        """
        Build weights in linear layer
        """

        name = f"linear_layer"
        self.weight = weight_initializer.glorot_uniform(obj_name=name, shape=[self.input_dimension, self.output_dimension])
        if self.use_bias:
            self.bias = weight_initializer.zeros(obj_name=name + "_bias", shape=[self.output_dimension])

    def bias_forward(self, input_data: tf.Tensor, weight: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Execute the forward path with bias.

        Args:
            input_data: tf.Tensor
            weight: Tuple[tf.Tensor, tf.Tensor]

        Returns:
            output: tf.Tensor
        """

        return self.activation(tf.matmul(input_data, weight[0]) + weight[1])

    def no_bias_forward(self, input_data: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
        """
        Execute the forward path without bias.

        Args:
            input_data: tf.Tensor
            weight: Tuple[tf.Tensor, tf.Tensor]
            
        Returns:
            output: tf.Tensor
        """

        return self.activation(tf.matmul(input_data, weight))

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Execute the forward path.

        Args:
            input_data: tf.Tensor
            
        Returns:
            output: tf.Tensor
        """

        if self.use_bias:
            return self.bias_forward(input_data, [self.weight, self.bias])
        else:
            return self.no_bias_forward(input_data, [self.weight])

