import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        # norm_in=True
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        # 检查是否对输入进行归一化
        if norm_in:  # normalize inputs
            # 创建一个一维的批量归一化层，用于对输入进行归一化
            self.in_fn = nn.BatchNorm1d(input_dim)
            # 初始化批量归一化层的权重和偏差参数，使其初始值为1和0，以确保输入数据的平均值接近0，方差接近1
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # 新加——层归一化

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # 新加——层归一化

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

        # 检查是否需要约束输出范围，并且动作空间是连续的
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            # 指定输出激活函数为双曲正切函数（tanh），以确保输出范围在[-1, 1]之间
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # h1 = self.nonlin(self.fc1(self.in_fn(X)))
        # h2 = self.nonlin(self.fc2(h1))

        # 改
        h1 = self.nonlin(self.ln1(self.fc1(self.in_fn(X))))
        h2 = self.nonlin(self.ln2(self.fc2(h1)))

        out = self.out_fn(self.fc3(h2))
        return out