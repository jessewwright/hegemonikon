import unittest
from nes.comparator import Comparator

class TestComparator(unittest.TestCase):
    def setUp(self):
        self.comparator = Comparator(drift_rate=0.3, threshold=1.0, noise_std=0.1)
    
    def test_initialization(self):
        self.assertEqual(self.comparator.drift_rate, 0.3)
        self.assertEqual(self.comparator.threshold, 1.0)
        self.assertEqual(self.comparator.noise_std, 0.1)
    
    def test_run_trial(self):
        # TODO: Add more specific test cases
        result = self.comparator.run_trial()
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
