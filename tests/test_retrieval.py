import unittest
from pathlib import Path

from rag import BankingKnowledgeBase


class BankingKnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge_base = BankingKnowledgeBase.from_json(
            Path(__file__).parents[1] / "data" / "banking_faq.json"
        )

    def test_retrieves_relevant_security_passage(self) -> None:
        results = self.knowledge_base.retrieve("How can I protect my online bank login?")

        self.assertTrue(results)
        self.assertEqual(results[0].title, "Keeping online banking accounts secure")
        self.assertGreater(results[0].score, 0)

    def test_empty_query_returns_no_results(self) -> None:
        self.assertEqual(self.knowledge_base.retrieve("  "), [])

    def test_invalid_top_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.knowledge_base.retrieve("bank account", top_k=0)


if __name__ == "__main__":
    unittest.main()
