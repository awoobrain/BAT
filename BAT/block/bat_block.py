from __future__ import annotations
from typing import Optional, NoReturn, Callable, List
from BAT.layer.attention_tournament_selection_layer   import AttentionTournamentSelectionLayer
from BAT.layer.feedforward_tournament_selection_layer import FeedForwardTournamentSelectionLayer
from BAT.component.layer_normalization import LayerNormalization
from ospark.nn.block import Block
import ospark
import tensorflow as tf


class BATEncoderBlock(Block):
    """
    Stack AttentionTournamentSelectionLayer and FeedForwardTournamentSelectionLayer in a block.
    """

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
				 head_number: int,
				 scale_rate: int,
				 dropout_rate: float,
                 att_conv_kernel_size: list,
				 ffnn_conv_kernel_size: int,
                 is_training: Optional[bool]=None
                 ):
        """
        Args:
            obj_name: str
            embedding_size: int
            head_number: int
            scale_rate: int 
            dropout_rate: float
            att_conv_kernel_size: list
            ffnn_conv_kernel_size: int 
            is_training: Optional[bool]
        """

        super().__init__(obj_name=obj_name, is_training=is_training)
        self._embedding_size        = embedding_size
        self._head_number           = head_number
        self._dropout_rate          = dropout_rate
        self._scale_rate            = scale_rate
        self._att_conv_kernel_size  = att_conv_kernel_size
        self._ffnn_conv_kernel_size = ffnn_conv_kernel_size

    @property
    def attention(self) -> AttentionTournamentSelectionLayer:
        return self._attention

    @property
    def feedforward(self) -> FeedForwardTournamentSelectionLayer:
        return self._feedforward

    def in_creating(self) -> NoReturn:
        """
        Build attention instance and feedforward instance.
        """
        
        self._attention = AttentionTournamentSelectionLayer(obj_name="attention",
                                                            embedding_size=self._embedding_size, 
                                                            head_number=self._head_number,
                                                            dropout_rate=self._dropout_rate,
                                                            conv_kernel_size=self._att_conv_kernel_size,
                                                            normalization1=LayerNormalization(layer_dimension=self._embedding_size, obj_name="LN1", use_bias=True, use_scale=True),
                                                            normalization2=LayerNormalization(layer_dimension=self._embedding_size, obj_name="LN2", use_bias=True, use_scale=True),
                                                            is_training=self.is_training)
        
        self._feedforward = FeedForwardTournamentSelectionLayer(obj_name="feedforward",
                                                                embedding_size=self._embedding_size, 
                                                                scale_rate=self._scale_rate,
                                                                dropout_rate=self._dropout_rate,
                                                                conv_kernel_size=self._ffnn_conv_kernel_size,
                                                                normalization=LayerNormalization(layer_dimension=self._embedding_size, obj_name="LN", use_bias=True, use_scale=True),
                                                                is_training=self.is_training)

    def pipeline(self, input_data: tf.Tensor, mask: Optional[tf.Tensor]=None) -> tf.Tensor:
        """
        Execute the forward path of the block.
        
        Args:
            input_data: tf.Tensor
            mask: Optional[tf.Tensor]
        
        Returns:
            output: tf.Tensor
        """

        output = self._attention.pipeline(input_data=input_data, mask=mask)
        output = self._feedforward.pipeline(input_data=output)
        return output