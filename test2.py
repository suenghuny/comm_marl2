import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = [nn.Linear(4, 4, bias=False) for _ in range(2)]
        self.W_q_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q])

# 테스트
model1 = TestModel()
print("Original weights:", model1.W_q[0].weight.data)

state_dict = model1.state_dict()
model2 = TestModel()
model2.load_state_dict(state_dict)

print("Loaded weights:", model2.W_q[0].weight.data)
print("Equal?:", torch.equal(model1.W_q[0].weight.data, model2.W_q[0].weight.data))