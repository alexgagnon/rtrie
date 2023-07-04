import pytest
from rtrie import StringTrie
from deepdiff import DeepDiff
        
words = ['Hello', 'Hey', 'Man', 'Man', 'Mani', 'Manilla', 'Manitee', 'There']
words = [(word, i) for i, word in enumerate(words)]

def test_string_trie():
    trie = StringTrie(words = iter(words))
    print(trie)