import unittest
from durbango.torch_utils import print_tensor_sizes, find_names
import torch

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(1,1)
    def forward(self, x):
        self.w(x)
        #self.log_mem(f'shape:{x.shape}')
        return x


class DummyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(3, 1)
        self.useless = Layer()

    def forward(self, x):
        output = self.w(x)
        return self.useless(output)

test_tensor = torch.tensor([1.,2., 3.]).unsqueeze(0)

class TestDebugTools(unittest.TestCase):


    def test_print_tensor_sizes(self):
        x = torch.tensor([1, 2, 3])
        results = print_tensor_sizes()

        expected_shape = (2,5)
        self.assertEqual(expected_shape, results.shape)

from durbango.logging_patch import patch_module_with_memory_mixin


class TestLoggingUtils(unittest.TestCase):

    def test_logger(self):
        model = DummyModule()
        model(test_tensor)
        #infect_module_(model)
        model.apply(patch_module_with_memory_mixin)
        model.reset_logs()
        model.log_mem()
        model(test_tensor)
        model.log_mem()
        log_df = model.combine_logs()
        print(model.summary)



