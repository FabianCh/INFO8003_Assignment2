import unittest
import Domain


class MyTestCase(unittest.TestCase):
    def DomainTest(self):
        domain = Domain.Domain()
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
