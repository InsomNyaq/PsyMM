import copy
import unittest
from psymm_generation.ola_inputs import fewshot_examples


class OlaInputContract(unittest.TestCase):
    def test_no_target_leakage_no_mutation_and_correct_example_history(self):
        def dialogue(cid, content, response):
            return dict(conversation_id=cid, turns=[dict(context='context '+cid,
                        dialogue_history=[dict(role='speaker',utterance=content)],response=response)])
        pool = [dialogue('target','SECRET','SECRET ANSWER'),
                dialogue('train1','example one','answer one'),
                dialogue('train2','example two','answer two')]
        before = copy.deepcopy(pool)
        prompt = fewshot_examples(pool, {'target'},2,13)
        self.assertNotIn('SECRET',prompt)
        self.assertIn('example one',prompt)
        self.assertIn('answer two',prompt)
        self.assertEqual(pool,before)
        self.assertEqual(prompt,fewshot_examples(pool,{'target'},2,13))
        with self.assertRaises(ValueError):
            fewshot_examples(pool,{'target','train1'},2,13)


if __name__ == '__main__':
    unittest.main()
