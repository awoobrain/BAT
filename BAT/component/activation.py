import tensorflow as tf 
from typing import NoReturn


class Activation:
    """
    Activation function: f()
    """
    
    def __init_subclass__(cls) -> NoReturn:
        super().__init_subclass__()
        setattr(Activation, cls.__name__, cls)
    
    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        run the f(x)

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
        Run calculate function.

        Args:
            input_data: tf.Tensor
        
        Returns:
            output: tf.Tensor
        """
        
        return self.calculate(input_data=input_data)


class PassActivation(Activation):
    """
    PassActivation formula is y = f(x) = x.
    """

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        run the f(x)

        Args:
            input_data: tf.Tensor
            
        Returns:
            output: tf.Tensor
        """

        return input_data


class Relu(Activation):
    """
    Relu formula is y = f(x) = max(0,x).
    """

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        run the f(x)

        Args:
            input_data: tf.Tensor
            
        Returns:
            output: tf.Tensor
        """

        return tf.nn.relu(input_data)


class Gelu(Activation):
    """
    Gelu formula is y = f(x) = xP(X<=x) ≈ 0.5x*(1+tanh[sqrt(2/pi)*(x+0.044715x^3)])
    """

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        run the f(x)

        Args:
            input_data: tf.Tensor
            
        Returns:
            output: tf.Tensor
        """

        return tf.nn.gelu(input_data)


class Softmax(Activation):
    """
    Softmax formula is yi = f(xi) = e^xi/(e^x1+e^x2+...e^xn)
    """

    def calculate(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        run the f(x)

        Args:
            input_data: tf.Tensor
            
        Returns:
            output: tf.Tensor
        """

        return tf.nn.softmax(input_data, axis=-1)