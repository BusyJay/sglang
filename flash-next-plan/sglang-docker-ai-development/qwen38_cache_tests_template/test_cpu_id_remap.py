import unittest


def remap(topk_ids, expert_to_slot):
    return [[expert_to_slot[expert] for expert in row] for row in topk_ids]


class TestIdRemap(unittest.TestCase):
    def test_repeated_expert_uses_same_slot(self):
        topk_ids = [[7, 20], [20, 91]]
        expert_to_slot = {7: 4, 20: 18, 91: 2}
        self.assertEqual(remap(topk_ids, expert_to_slot), [[4, 18], [18, 2]])


if __name__ == "__main__":
    unittest.main()
