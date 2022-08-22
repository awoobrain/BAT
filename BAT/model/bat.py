from BAT.component.layer_normalization import LayerNormalization
from BAT.block.bat_block import BATEncoderBlock
from BAT.layer.dense_layer import DenseLayer
from ospark import Model 
from typing import List, NoReturn, Optional
import tensorflow as tf
import numpy as np

class BAT(Model):
    """
    Build BAT model.
    """

    def __init__(self, 
                 obj_name: str, 
                 block_number: int,
                 embedding_size: int, 
                 head_number: int, 
                 scale_rate: int,
                 att_conv_kernel_size: List[int],
                 ffnn_conv_kernel_size: int,
                 dropout_rate: float,
                 classify_num: int,
                 is_training: Optional[bool]=None,
                 max_sequence_length: Optional[int]=None,
                 trained_weights: Optional[dict]=None):
        """
        Args:
            obj_name: str
            block_number: int
            embedding_size: int
            head_number: int
            scale_rate: int
            att_conv_kernel_size: List[int]
                The kernel size in convolution1 and convolution2.
            ffnn_conv_kernel_size: int
                The kernel size in ffnn convolution.
            dropout_rate: float
            classify_num: int
            is_training: Optional[bool]
            max_sequence_length: Optional[int]
                Decide positional encoding length.
                default: 2000
            trained_weights: Optional[dict]
        """

        super().__init__(obj_name=obj_name, trained_weights=trained_weights, is_training=is_training)
        self._embedding_size        = embedding_size
        self._head_number           = head_number
        self._dropout_rate          = dropout_rate
        self._scale_rate            = scale_rate
        self._att_conv_kernel_size  = att_conv_kernel_size
        self._ffnn_conv_kernel_size = ffnn_conv_kernel_size
        self._block_number          = block_number
        self._encoders              = []
        self._classify_layer        = None
        self._classify_num          = classify_num
        self._max_sequence_length   = max_sequence_length or 2000

        self._positional_encoding_table = self.create_positional_encoding_table()

    def in_creating(self) -> NoReturn:
        """
        Build module in BAT.
        """

        for i in range(self._block_number):
            encoder_name = f"encoder_{i}"

            encoder = BATEncoderBlock(obj_name=encoder_name, 
                                      embedding_size=self._embedding_size,
                                      head_number=self._head_number,
                                      scale_rate=self._scale_rate,
                                      dropout_rate=self._dropout_rate,
                                      att_conv_kernel_size=self._att_conv_kernel_size,
                                      ffnn_conv_kernel_size=self._ffnn_conv_kernel_size,
                                      is_training=self.is_training)
            self._encoders.append(encoder)
        self._classify_layer = DenseLayer(obj_name="classify_layer", 
                                          input_dimension=self._embedding_size,
                                          output_dimension=self._classify_num,
                                          activation="Softmax", 
                                          use_bias=True)
        
        self.layer_normalization = LayerNormalization(layer_dimension=self._embedding_size,
                                                      obj_name="LN_after_embedding", 
                                                      use_bias=True, 
                                                      use_scale=True)
    
    def pipeline(self, 
                 input_data: tf.Tensor, 
                 encoder_padding_mask: Optional[tf.Tensor]=None, 
                 use_positional_encoding: Optional[bool]=None) -> tf.Tensor:
        """
        Execute the forward path of the model.
        
        Args:
            input_data: tf.Tensor
                shape:[batch,seq_len,embed_dim]
            encoder_padding_mask: Optional[tf.Tensor]
                padding position set one, the others get set zero
            use_positional_encoding: Optional[bool]
                defualt: True
        
        Returns:
            output: tf.Tensor
        """

        use_positional_encoding  = True if use_positional_encoding is None else use_positional_encoding
        if encoder_padding_mask == None:
            encoder_padding_mask = self.create_embedding_mask(input_data)
        
        seq_len = tf.shape(input_data)[1]
        # print("position",self._positional_encoding_table[:, :seq_len, :].shape)
        output = input_data
        if use_positional_encoding:
            output += self._positional_encoding_table[:, :seq_len, :]
        output = self.layer_normalization(output)
        for encoder_block in self._encoders:
            output = encoder_block.pipeline(input_data=output,
                                            mask=encoder_padding_mask)
        
        output = self._classify_layer.pipeline(output)
        return output

    def create_embedding_mask(self, encoder_input: tf.Tensor) -> tf.Tensor:
        """
        Add padding mask. 
            If the encoder_input is zero -> mask get one
            If the encoder_input is others -> mask get zero
            
        Args:
            encoder_input: tf.Tensor
                shape:[batch,seq_len,embed_dim]
        
        Returns:
            encoder_padding_mask: tf.Tensor
        """

        encoder_padding_mask = tf.cast(tf.math.equal(encoder_input[:, :, 0], 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        print("內建mask", encoder_padding_mask.shape)
        return encoder_padding_mask


    def create_positional_encoding_table(self) -> tf.Tensor:
        """
        Create positional encoding.

        Returns:
            output: tf.Tensor
        """

        basic_table = np.zeros(shape=[self._max_sequence_length, self._embedding_size])
        position    = np.arange(self._max_sequence_length).reshape([-1, 1])
        denominator = np.power(10000, np.arange(0, self._embedding_size, 2) / self._embedding_size)
        
        basic_table[:, 0::2] = np.sin(position / denominator)
        basic_table[:, 1::2] = np.cos(position / denominator)
        return tf.convert_to_tensor(basic_table, dtype=tf.float32)[tf.newaxis, :, :]

