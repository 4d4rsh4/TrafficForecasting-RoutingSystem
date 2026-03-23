import unittest
import torch
from app import TemporalGCNGRU


class TestGCNGRU(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.seq_in = 12
        self.num_sensors = 307
        self.num_features = 3
        self.seq_out = 12
        self.adj = torch.eye(self.num_sensors)
        self.model = TemporalGCNGRU(
            num_sensors=self.num_sensors, gcn_hid=32, gru_hid=64,
            seq_in=self.seq_in, seq_out=self.seq_out, adj=self.adj
        )
        self.dummy_input = torch.randn(
            self.batch_size, self.seq_in, self.num_sensors, self.num_features
        )

    def test_model_output_shape(self):
        output = self.model(self.dummy_input)
        expected_shape = (self.batch_size, self.seq_out, self.num_sensors)
        self.assertEqual(output.shape, expected_shape)
        print("Unit Test Passed: Output shape is correct:", output.shape)


if __name__ == '__main__':
    unittest.main()