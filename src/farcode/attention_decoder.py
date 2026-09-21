# @Time    : 2019/05/13 15:29
# @Author  : niuliangtao
# @Site    :
# @File    : temp2.py
# @Software: PyCharm

from typing import Any

import tensorflow as tf
from farlog import getLogger
from keras import activations, constraints, initializers, regularizers
from keras.engine import InputSpec
from keras.layers.recurrent import Recurrent

logger = getLogger(__name__)

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

"""实现时间维度上的稠密层，改编自 Keras 后端。"""
import keras.backend as K


def _time_distributed_dense(
    x,
    w,
    b=None,
    dropout=None,
    input_dim=None,
    output_dim=None,
    timesteps=None,
    training=None,
):
    """对输入的每个时间片应用 ``y @ w + b``。"""
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0.0 < dropout < 1.0:
        # 每个时间步使用相同的 dropout 掩码。
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # 合并时间维度和批次维度。
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # 恢复为三维张量。
    if K.backend() == "tensorflow":
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class AttentionDecoder(Recurrent):
    def __init__(
        self,
        units: int,
        output_dim: int,
        activation: str = "tanh",
        return_probabilities: bool = False,
        name: str = "AttentionDecoder",
        kernel_initializer: Any = "glorot_uniform",
        recurrent_initializer: Any = "orthogonal",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Any = None,
        bias_regularizer: Any = None,
        activity_regularizer: Any = None,
        kernel_constraint: Any = None,
        bias_constraint: Any = None,
        **kwargs,
    ):
        """创建将编码序列解码为输出序列的注意力层。

        Args:
            units: 隐状态和注意力矩阵的维度。
            output_dim: 输出标签空间的维度。
            activation: 隐状态激活函数名称。
            return_probabilities: 是否返回注意力概率。
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # 必须返回序列。

    def build(self, input_shape: Any) -> None:
        """根据输入形状创建 Bahdanau 2014 附录 2 中的权重矩阵。"""

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super().reset_states()

        self.states = [None, None]  # y, s

        # 用于计算上下文向量的矩阵。

        self.V_a = self.add_weight(
            shape=(self.units,),
            name="V_a",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.W_a = self.add_weight(
            shape=(self.units, self.units),
            name="W_a",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.U_a = self.add_weight(
            shape=(self.input_dim, self.units),
            name="U_a",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.b_a = self.add_weight(
            shape=(self.units,),
            name="b_a",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
        # 重置门矩阵。
        self.C_r = self.add_weight(
            shape=(self.input_dim, self.units),
            name="C_r",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.U_r = self.add_weight(
            shape=(self.units, self.units),
            name="U_r",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.W_r = self.add_weight(
            shape=(self.output_dim, self.units),
            name="W_r",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.b_r = self.add_weight(
            shape=(self.units,),
            name="b_r",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        # 更新门矩阵。
        self.C_z = self.add_weight(
            shape=(self.input_dim, self.units),
            name="C_z",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.U_z = self.add_weight(
            shape=(self.units, self.units),
            name="U_z",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.W_z = self.add_weight(
            shape=(self.output_dim, self.units),
            name="W_z",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.b_z = self.add_weight(
            shape=(self.units,),
            name="b_z",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
        # 候选隐状态矩阵。
        self.C_p = self.add_weight(
            shape=(self.input_dim, self.units),
            name="C_p",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.U_p = self.add_weight(
            shape=(self.units, self.units),
            name="U_p",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.W_p = self.add_weight(
            shape=(self.output_dim, self.units),
            name="W_p",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.b_p = self.add_weight(
            shape=(self.units,),
            name="b_p",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )
        # 最终预测向量矩阵。
        self.C_o = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            name="C_o",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.U_o = self.add_weight(
            shape=(self.units, self.output_dim),
            name="U_o",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.W_o = self.add_weight(
            shape=(self.output_dim, self.output_dim),
            name="W_o",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.b_o = self.add_weight(
            shape=(self.output_dim,),
            name="b_o",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        # 创建初始状态。
        self.W_s = self.add_weight(
            shape=(self.input_dim, self.units),
            name="W_s",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))
        ]
        self.built = True

    def call(self, x: Any) -> Any:
        # 保存完整序列，以便在每个时间步计算注意力。
        self.x_seq = x

        # 时间维度上的稠密变换不依赖之前的步骤，因此提前计算以节省开销。
        self._uxpb = _time_distributed_dense(
            self.x_seq,
            self.U_a,
            b=self.b_a,
            input_dim=self.input_dim,
            timesteps=self.timesteps,
            output_dim=self.units,
        )

        return super().call(
            x,
        )

    def get_initial_state(self, inputs: Any) -> list[Any]:
        """根据输入序列计算初始输出和隐状态。"""
        logger.debug(f"inputs shape: {inputs.get_shape()}")

        # 使用第一个时间步计算初始隐状态。
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # 创建形状为 (batch_size, output_dim) 的初始输出向量。
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x: Any, states: list[Any]) -> tuple[Any, list[Any]]:
        """执行一个时间步并返回输出与下一状态。"""

        ytm, stm = states

        # 将隐状态重复到输入序列长度。
        _stm = K.repeat(stm, self.timesteps)

        # 将权重矩阵乘以重复后的隐状态。
        _Wxstm = K.dot(_stm, self.W_a)

        # 计算注意力概率，表示其他时间步对当前时间步的贡献。
        et = K.dot(activations.tanh(_Wxstm + self._uxpb), K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # 形状为 (batch_size, timesteps, 1)。

        # 计算上下文向量。
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # 计算新的隐状态，先计算重置门。

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r
        )

        # 计算更新门。
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z
        )

        # 计算候选隐状态。
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p
        )

        # 合成新的隐状态。
        st = (1 - zt) * stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o
        )

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape: Any) -> tuple[Any, ...]:
        """返回 Keras 内部兼容性检查所需的输出形状。"""
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self) -> dict[str, Any]:
        """返回用于重新加载模型的配置。"""
        config = {
            "output_dim": self.output_dim,
            "units": self.units,
            "return_probabilities": self.return_probabilities,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# 简单构建检查。
if __name__ == "__main__":
    from keras.layers import LSTM, Input
    from keras.layers.wrappers import Bidirectional
    from keras.models import Model

    i = Input(shape=(100, 104), dtype="float32")
    enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode="concat")(i)
    dec = AttentionDecoder(32, 4)(enc)
    model = Model(inputs=i, outputs=dec)
    model.summary()
