import torch.nn as nn
import torch

# 实例化RNN对象
# 第一个参数：input_size(输入张量x的维度)
rnn = nn.RNN(5, 6, 2)
input = torch.randn(1, 3, 5)
h0 = torch.randn(2, 3, 6)
output, hn = rnn(input,h0)
# 打印参数
print(output.shape)
print(hn.shape)
