import unittest
from durbango.torch_utils import print_tensor_sizes, find_names
import torch




class TestDebugTools(unittest.TestCase):


    def test_print_tensor_sizes(self):
        x = torch.tensor([1, 2, 3])
        results = print_tensor_sizes()
        #expected = (3,)
        #self.assertEqual(expected, results['x'])



