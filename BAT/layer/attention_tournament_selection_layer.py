import tensorflow as tf
from ospark.nn.layers import Layer
from ospark.nn.layers.normalization import Normalization
from BAT.component.activation import Activation
from BAT.component.layer_normalization import LayerNormalization
from ospark import weight_initializer
from typing import *
import ospark
import time


class AttentionTournamentSelectionLayer(Layer):
    """
    Contains a multi-feature self-attention block (MFSA) and a residual connection in parallel.
    """

    def __init__(self, 
                 obj_name: str, 
                 embedding_size: int, 
                 head_number: int,
                 dropout_rate: float,
                 is_training: Optional[bool]=None,
                 normalization1: Optional[Normalization]=None,
                 normalization2: Optional[Normalization]=None,
                 conv_kernel_size: Optional[List[int]]=None,
                 activation: Optional[Activation]=None):
        """
        Args:
            obj_name: str
            embedding_size: int
            head_number: int
            dropout_rate: float
            is_training: Optional[bool]
            normalization1: Optional[Normalization]
                default: LayerNormalization(layer_dimension, obj_name)
            normalization2: Optional[Normalization]
                default: LayerNormalization(layer_dimension, obj_name)
            conv_kernel_size: Optional[List[int]]
                The kernel size in convolution1 and convolution2.
                default: [1,3]
            activation: Optional[Activation] 
                deefault: Activation.Relu()
        """

        super().__init__(obj_name=obj_name, is_training=is_training)
        assert embedding_size % head_number == 0

        self._depth            = embedding_size // head_number
        self._embedding_size   = embedding_size
        self._head_number      = head_number
        self._sequence_length  = None
        self._normalization1   = normalization1 or LayerNormalization(layer_dimension=embedding_size, obj_name="LN1")
        self._normalization2   = normalization2 or LayerNormalization(layer_dimension=embedding_size, obj_name="LN2")
        self._dropout_layer    = tf.keras.layers.Dropout(rate=dropout_rate)
        self._conv_kernel_size = conv_kernel_size or [1,3]
        self._activation       = activation or Activation.Relu()

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def activation(self) -> Activation:
        return self._activation

    @property
    def head_number(self) -> int:
        return self._head_number

    @property
    def conv_kernel_size(self) -> list:
        return self._conv_kernel_size

    @property
    def normalization1(self) -> Normalization:
        return self._normalization1

    @property
    def normalization2(self) -> Normalization:
        return self._normalization2

    @property
    def dropout_layer(self) -> tf.keras.layers.Dropout:
        return self._dropout_layer

    def in_creating(self) -> NoReturn:
        """
        Build weights in ATS(AttentionTournamentSelection) layer
        """

        self._q_weights = weight_initializer.glorot_uniform(
            obj_name="q_weights",
            shape=[self.embedding_size, self.embedding_size]
        )
        self._k_weights = weight_initializer.glorot_uniform(
            obj_name="k_weights",
            shape=[self.embedding_size, self.embedding_size]
        )
        self._v_weights = weight_initializer.glorot_uniform(
            obj_name="v_weights",
            shape=[self.embedding_size, self.embedding_size]
        )
        self._q_bias = weight_initializer.zeros(
            obj_name="q_bias",
            shape=[self.embedding_size]
        )
        self._k_bias = weight_initializer.zeros(
            obj_name="k_bias",
            shape=[self.embedding_size]
        )
        self._v_bias = weight_initializer.zeros(
            obj_name="v_bias",
            shape=[self.embedding_size]
        )
        self._output_weights = weight_initializer.glorot_uniform(
            obj_name="output_weights",
            shape=[self.embedding_size, self.embedding_size]
        )
        self._output_bias = weight_initializer.zeros(
            obj_name="output_bias",
            shape=[self.embedding_size]
        )
        self._conv1 = weight_initializer.glorot_uniform(
            obj_name="conv1",
            shape=[self.conv_kernel_size[0], self.embedding_size, self.embedding_size]
        )
        self._conv1_bias = weight_initializer.zeros(
            obj_name="conv1_bias",
            shape=[self.embedding_size]
        )
        self._conv2 = weight_initializer.glorot_uniform(
            obj_name="conv2",
            shape=[self.conv_kernel_size[1], self.embedding_size, self.embedding_size]
        )
        self._conv2_bias = weight_initializer.zeros(
            obj_name="conv2_bias",
            shape=[self.embedding_size]
        )
        self._norm1 = self.normalization1
        self._norm2 = self.normalization2

    def pipeline(self, input_data: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        """
        Execute the forward path of the layer.
        
        Args:
            input_data: tf.Tensor
            mask: Optional[tf.Tensor]
                Used in attention structure.
        
        Returns:
            output: tf.Tensor
        """

        return self.layer_calculation(Q_input=input_data,
                                      K_input=input_data,
                                      V_input=input_data,
                                      mask=mask)

    def layer_calculation(self,
                          Q_input: tf.Tensor,
                          K_input: tf.Tensor,
                          V_input: tf.Tensor,
                          mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Execute the forward path of the layer.
        
        Args:
            Q_input: tf.Tensor
            K_input: tf.Tensor
            V_input: tf.Tensor
            mask: Optional[tf.Tensor]
                Used in attention structure.
        
        Returns:
            layer_output: tf.Tensor
        """

        batch_size = tf.shape(Q_input)[0]

        Q, K, V      = self.QKV_process(Q_input=Q_input, K_input=K_input, V_input=V_input, batch_size=batch_size)
        main_output  = self.attention_layer(Q=Q, K=K, V=V, batch_size=batch_size, mask=mask)
        main_output  = self.dropout_layer(main_output, training=self.is_training)
        conv1_output = tf.nn.conv1d(input=Q_input, 
                                    filters=self._conv1,
                                    stride=[1,1,1],
                                    padding="SAME")
        conv1_output = conv1_output + self._conv1_bias
        conv1_output = self.activation(conv1_output)
        conv2_output = tf.nn.conv1d(input=Q_input, 
                                    filters=self._conv2,
                                    stride=[1,1,1],
                                    padding="SAME")
        conv2_output    = conv2_output + self._conv2_bias
        conv2_output    = self.activation(conv2_output)
        layer_output    = self._norm1(tf.math.add_n([conv1_output, main_output, conv2_output]))
        residual_output = self.residual_net(input_data=Q_input)
        layer_output    = self._norm2(tf.math.add(layer_output, residual_output))
        return layer_output

    def QKV_process(self,
                    Q_input: tf.Tensor,
                    K_input: tf.Tensor,
                    V_input: tf.Tensor,
                    batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Transfer the Q_input,K_input,V_input into new space and split the embedding_size into (head_number,depth) shape.
        
        Args:
            Q_input: tf.Tensor
            K_input: tf.Tensor
            V_input: tf.Tensor
            batch_size: tf.Tensor
        
        Returns:
            output: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        """

        Q = tf.matmul(Q_input, self._q_weights) + self._q_bias  # [batch, seq, d_model]
        K = tf.matmul(K_input, self._k_weights) + self._k_bias
        V = tf.matmul(V_input, self._v_weights) + self._v_bias
        Q = self.split_head(input_data=Q, batch_size=batch_size) # [batch, head_number, seq, depth]
        K = self.split_head(input_data=K, batch_size=batch_size)
        V = self.split_head(input_data=V, batch_size=batch_size)
        return Q, K, V

    def split_head(self, input_data: tf.Tensor, batch_size: tf.int32) -> tf.Tensor:
        """
        Split the embedding_size into (head_number,depth) shape.
        
        Args:
            input_data: tf.Tensor
            batch_size: tf.int32
        
        Returns:
            output: tf.Tensor
        """

        split_result     = tf.reshape(input_data, [batch_size, -1, self.head_number, self.depth]) # [batch, seq, head_number, depth]
        transpose_result = tf.transpose(split_result, [0, 2, 1, 3]) # [batch, head_number, seq, depth]
        return transpose_result

    def attention_layer(self, 
                        Q: tf.Tensor, 
                        K: tf.Tensor, 
                        V: tf.Tensor,
                        batch_size: tf.Tensor,
                        mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        """
        Execute attention calculation.
        
        Args:
            Q: tf.Tensor
            K: tf.Tensor
            V: tf.Tensor
            batch_size: tf.Tensor
            mask: Optional[tf.Tensor]
        
        Returns:
            layer_output: tf.Tensor
        """

        attention_value = self.attention(Q=Q, K=K, V=V, batch_size=batch_size, mask=mask)
        layer_output    = tf.matmul(attention_value, self._output_weights) + self._output_bias
        return layer_output

    def attention(self,
                  Q: tf.Tensor,
                  K: tf.Tensor,
                  V: tf.Tensor,
                  batch_size: tf.Tensor,
                  mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        """
        Part of attention calculation.
        
        Args:
            Q: tf.Tensor
            K: tf.Tensor
            V: tf.Tensor
            batch_size: tf.Tensor
            mask: Optional[tf.Tensor]
        
        Returns:
            concat_output: tf.Tensor
        """
        
        K = tf.transpose(K, [0, 1, 3, 2]) # BHLD -> BHDL
        scaled_dot_product = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.embedding_size, dtype=tf.float32)) # BHLD * BHDL -> BHLL
        if mask is not None:
            scaled_dot_product += (mask * -1e9)
        scaled_dot_product = tf.nn.softmax(scaled_dot_product, axis=-1)
        scaled_attention   = tf.matmul(scaled_dot_product, V) # BHLL * BHLD -> BHLD
        scaled_attention   = tf.transpose(scaled_attention, [0, 2, 1, 3]) # BHLD -> BLHD
        concat_output      = tf.reshape(scaled_attention, [batch_size, -1, self.embedding_size])
        return concat_output

    def residual_net(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Transmission of input data

        Args:
            input_data: tf.Tensor
        
        Returns:
            input_data: tf.Tensor
        """

        return input_data