import pytest
import unittest

from agent.global_agent import *

# Testing: Incomplete

# RAG testing:
# 1. Test the keyword search function (search_conversations)
# 2. Test the chaching function (write_cache)

# Possible tests:
# 1. Type test
# 2. Operational errors

class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = AIAgent()

    # search_conversations test
    def test_search_conversations_non_string_input(self):
        # Make sure that the function doesn't accept numerical input during keyword search
        with self.assertRaises(TypeError):
            search_conversations(['depression', 'anxiety', 1, 2.8, 3 + 8j])

    def test_search_conversations_empty_string_input(self):
        # Make sure that the function doesn't accept empty string input during keyword search
        with self.assertRaises(ValueError):
            search_conversations(['', 'help', 'stress'])
    
    # write_cache test
    def test_write_cache_string_index(self):
        # Make sure that the function doesn't accept string values as indices during cache writing
        with self.assertRaises(TypeError):
            write_cache(['hello', 'world', 0, 3, 4])

    def test_write_cache_empty_index(self):
        # Make sure that the function doesn't accept an empty list as the list of indices
        with self.assertRaises(ValueError):
            write_cache([])

    
    # agent behaviour test
    def test_agent_getintent_nonstring(self):
        # Make sure that the function doesn't accept non-string input during intent identification
        with self.assertRaises(TypeError):
            self.agent.getIntent(123)
        
        with self.assertRaises(TypeError):
            self.agent.getIntent([1, 2, 3])
        
        with self.assertRaises(TypeError):
            self.agent.getIntent(0.97)

        with self.assertRaises(TypeError):
            self.agent.getIntent(None)

        with self.assertRaises(TypeError):
            self.agent.getIntent(True)

        with self.assertRaises(TypeError):
            self.agent.getIntent(False)

        with self.assertRaises(TypeError):
            self.agent.getIntent(1 + 3j)

    


if __name__ == '__main__':
    pytest.main(args = [''])