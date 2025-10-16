import unittest
from afd_ami_core import AFDInfinityAMI

class TestAFDInfinityAMI(unittest.TestCase):
    def setUp(self):
        self.assistant = AFDInfinityAMI()

    def test_coherence_score(self):
        state = [0.5, 0.5, 10]  # [sentiment, coherence_prev, length]
        action = [0.5, 0.5]     # [sentiment_impact, coherence_impact]
        score, _ = self.assistant.coherence_score(action, state)
        self.assertTrue(0 <= score <= 1, "Coherence score should be between 0 and 1")

    def test_reflect_ethics_adjustment(self):
        self.assistant.memory_scores = [0.6] * 6  # Simulate low coherence
        reflection = self.assistant.reflect_ethics()
        self.assertTrue(self.assistant.alpha > 1.0, "Alpha should increase for low coherence")
        self.assertIn("Adjusted", reflection)

    def test_memory_saving(self):
        prompt, response = "Test", "Test response"
        self.assistant.respond(prompt)
        self.assistant.save_memory(prompt, response, 0.8)
        self.assertTrue(os.path.exists('data/response_log.csv'))

if __name__ == '__main__':
    unittest.main()
