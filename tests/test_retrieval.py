import unittest

from banking_assistant.retrieval import load_knowledge_base, retrieve


class RetrievalTests(unittest.TestCase):
    def test_bundled_knowledge_base_has_articles(self):
        articles = load_knowledge_base()
        self.assertGreaterEqual(len(articles), 5)
        self.assertTrue(all(article.get("source") for article in articles))

    def test_query_returns_relevant_article_first(self):
        results = retrieve("What happens if I see a suspicious transaction?")
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "fraud-response")
        self.assertGreater(results[0]["score"], 0)

    def test_unrelated_query_has_no_matches(self):
        self.assertEqual(retrieve("Tell me about tropical fruit recipes"), [])

    def test_blank_query_has_no_matches(self):
        self.assertEqual(retrieve("   "), [])

    def test_limit_must_be_positive(self):
        with self.assertRaises(ValueError):
            retrieve("checking account", limit=0)


if __name__ == "__main__":
    unittest.main()
