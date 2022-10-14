import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)

PATH = "mnist_ffn.pth"
model.load_state_dict(torch.load(PATH))
model.eval()