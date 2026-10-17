import os
import unittest
from unittest.mock import patch

from banking_assistant.generation import llm_is_configured, st


class GenerationConfigurationTests(unittest.TestCase):
    def test_environment_api_key_enables_generation(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            self.assertTrue(llm_is_configured())

    def test_blank_environment_key_does_not_enable_generation(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.assertFalse(llm_is_configured())

    def test_streamlit_secret_enables_generation(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
            patch.object(st, "secrets", {"OPENAI_API_KEY": "deployment-key"}),
        ):
            self.assertTrue(llm_is_configured())


if __name__ == "__main__":
    unittest.main()
