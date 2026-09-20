# 文件名: my_embedding_net.py
# 位置: ~/桌面/descriptor/my_embedding_net.py

import torch.nn as nn
from deepmd.dpmodel.descriptor.se_e2_a import SeE2A
from deepmd.dpmodel.descriptor.base_descriptor import BaseDescriptor

# 注册描述符类型，名称为 "my_embedding_net"
@BaseDescriptor.register("my_embedding_net")
class MyEmbeddingNet(SeE2A):
    """
    自定义描述符：继承自标准 se_e2_a，但使用用户定义的多层嵌入网络。
    实现了“多层原子嵌入”设计，您可在此基础上扩展“耦合”机制。
    """
    def __init__(
        self,
        rcut: float = 6.0,
        neuron: list = [256, 128, 64],   # 嵌入网络神经元结构
        **kwargs
    ):
        # 调用父类初始化，父类会基于默认参数构建嵌入网络
        super().__init__(rcut=rcut, **kwargs)
        # 保存自定义神经元配置
        self.neuron = neuron
        # 用自定义的嵌入网络替换父类中的 embedding_net
        self.embedding_net = self._build_embedding_net()

    def _build_embedding_net(self):
        """
        构建自定义的多层全连接网络。
        输入维度为 1（因为 se_e2_a 的嵌入网络输入是径向距离，标量）
        输出维度为 neuron[-1]
        """
        layers = []
        in_dim = 1  # 输入为距离（标量）
        for out_dim in self.neuron:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())  # 使用 SiLU (Swish) 激活
            in_dim = out_dim
        # 返回一个 Sequential 容器
        return nn.Sequential(*layers)

    # 其他方法（get_rcut, get_sel, get_dim 等）均继承自父类，无需重写
