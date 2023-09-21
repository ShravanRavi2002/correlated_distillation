import torch

torch.set_default_dtype(torch.float64)
class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)

     def forward(self, x):
         outputs = self.linear(x)
         return outputs