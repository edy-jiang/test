# import torch.nn as nn
import torch
import torch.nn as nn
rnn = nn.LSTM(5, 6, 1,bidirectional=True)


# ---->input(sequence_length, batch, input_size)
input = torch.randn(1, 3, 5)
# ---->hidden_size = hidden_size//2
# ---->h0(num_layers*num_directions, batch, hidden_size)
h0 = torch.randn(2, 3, 6)
# ---->h0(num_layers*num_directions, batch, hidden_size)
c0 = torch.randn(2, 3, 6)
output, (hn, cn) = rnn(input, (h0, c0))
# output, hn = rnn(input , h0)
# 注意，output的结果与hn的最后一层的结果是一模一样的
# 因为num_layer当前设置的为2，当前hn有两个输出结果

print(output)
print("output.shape是",output.shape)
# ---->output.shape是 torch.Size([1, 3, 12])
# ---->output的前两维一定和input一样
print(hn)
print("hn.shape是",hn.shape)
# ---->hn.shape是 torch.Size([2, 3, 6])
# ---->同h0维度一样(num_layers*num_directions, batch, hidden_size)
print(cn)
print("cn.shape是",cn.shape)
# ---->cn.shape是 torch.Size([2, 3, 6])