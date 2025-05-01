import unittest
from nes.norm_repository import NormRepository

class TestNormRepository(unittest.TestCase):
    def setUp(self):
        self.repo = NormRepository()
    
    def test_initialization(self):
        self.assertIsNotNone(self.repo)
        self.assertEqual(len(self.repo.norms), 0)
    
    def test_add_and_retrieve(self):
        # TODO: Add more specific test cases
        self.repo.add_norm("test_norm")
        result = self.repo.retrieve_norms("test_context")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
