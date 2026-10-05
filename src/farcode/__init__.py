"""farfarfun 深度学习与数据处理代码合集。

当前可用的只有 :mod:`farcode.data`——UCI Adult 收入数据集与亚马逊 Electronics
评论数据集的下载/预处理脚本。

:mod:`farcode.attention_decoder` 是 2019 年基于 Keras 2.2 + TensorFlow 1.x 写的
历史参考代码，在 ``requires-python >= 3.12`` 下没有任何可安装的依赖组合能让它运行，
因此不在这里导入，也不属于本包的公开 API；直接导入它会得到一条说明现状的
``ImportError``。
"""
