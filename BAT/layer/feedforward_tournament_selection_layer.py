from ospark.nn.layers import Layer
from ospark.nn.layers.normalization import Normalization
from ospark import weight_initializer
from BAT.component.activation import Activation
from BAT.component.layer_normalization import LayerNormalization
from typing import NoReturn, Optional
import tensorflow as tf 
import ospark


class FeedForwardTournamentSelectionLayer(Layer):
    """
    FFTS, Contains FFNN and weighted residual connection.
    """

    def __init__(self, 
                 obj_name: str, 
                 embedding_size: int, 
                 scale_rate: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=None,
                 activation: Optional[Activation]=None,
                 normalization: Optional[Normalization]=None,
                 conv_kernel_size: Optional[int]=None) -> NoReturn:
        """
        Args:
            obj_name: str
            embedding_size: int
            scale_rate: int
            dropout_rate: float
            is_training: Optional[bool]
            activation: Optional[Activation]
                default: Activation.Relu()
            normalization: Optional[Normalization]
                default: LayerNormalization(layer_dimension, obj_name)
            conv_kernel_size: Optional[int]
                default: 1
        """

        super().__init__(obj_name=obj_name, is_training=is_training)
        self._normalization    = normalization or LayerNormalization(layer_dimension=embedding_size, obj_name="LN")
        self._activation       = activation or Activation.Relu()
        self._embedding_size   = embedding_size
        self._scale_rate       = scale_rate
        self._dropout_layer    = tf.keras.layers.Dropout(rate=dropout_rate)
        self._conv_kernel_size = conv_kernel_size or 1

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def scale_rate(self) -> int:
        return self._scale_rate

    @property
    def activation(self) -> Activation:
        return self._activation

    @property
    def normalization(self) -> Normalization:
        return self._normalization

    @property
    def dropout_layer(self) -> tf.keras.layers.Dropout:
        return self._dropout_layer

    @property
    def conv_kernel_size(self) -> int:
        return self._conv_kernel_size

    def in_creating(self) -> NoReturn:
        """
        Build weights in FFTS.
        """

        self._mapping2high_dimensional = weight_initializer.glorot_uniform(
            obj_name="mapping2high_dimensional", 
            shape=[self.embedding_size, self.scale_rate * self.embedding_size]
        )
        self._mapping2low_dimensional = weight_initializer.glorot_uniform(
            obj_name="mapping2low_dimensional", 
            shape=[self.scale_rate * self.embedding_size, self.embedding_size]
        )
        self._high_dimensional_bias = weight_initializer.zeros(
            obj_name="high_dimensional_bias",
            shape=[self.scale_rate * self.embedding_size]
        )
        self._low_dimensional_bias = weight_initializer.zeros(
            obj_name="low_dimensional_bias",
            shape=[self.embedding_size]
        )
        self._conv = weight_initializer.glorot_uniform(
            obj_name="conv",
            shape=[self.conv_kernel_size, self.embedding_size, self.embedding_size]
        )
        self._conv_bias = weight_initializer.zeros(
            obj_name="conv_bias",
            shape=[self.embedding_size]
        )
        self._norm = self.normalization

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Execute the forward path of the layer.
        
        Args:
            input_data: tf.Tensor
        
        Returns:
            normalization_output: tf.Tensor
        """

        main_output          = self.feedforward(input_data)
        residual_output      = self.weighted_residual_net(input_data)
        added_residual       = tf.add(self.dropout_layer(main_output, training=self.is_training), residual_output)
        normalization_output = self._norm(added_residual)
        return normalization_output

    def feedforward(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        FFNN the same as Bert.
        
        Args:
            input_data: tf.Tensor
        
        Returns:
            mapping2low_dimensional: tf.Tensor
        """

        mapping2high_dimensional = tf.matmul(input_data, self._mapping2high_dimensional) + self._high_dimensional_bias
        activated_outputs        = self.activation(mapping2high_dimensional)
        mapping2low_dimensional  = tf.matmul(activated_outputs, self._mapping2low_dimensional) + self._low_dimensional_bias
        return mapping2low_dimensional

    def weighted_residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Do a transformation in residual.

        Args:
            input_data: tf.Tensor
        
        Returns:
            conv_output: tf.Tensor
        """

        conv_output = tf.nn.conv1d(input=input_data, 
                                    filters=self._conv,
                                    stride=[1,1,1],
                                    padding="SAME")
        conv_output = conv_output + self._conv_bias               
        conv_output = self.activation(conv_output)
        return conv_output