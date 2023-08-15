from rtrie import Trie
import logging
import os
from rtrie.array_trie import ArrayTrie
from rtrie.string_trie import StringTrie

level = logging.DEBUG if os.environ.get("DEBUG") == "True" else logging.INFO
logging.basicConfig(level = level)

words = ['Hello', 'Hey', 'Man', 'Man', 'Mani', 'Manilla', 'Manitee', 'Q', 'There']
words = [(word, i) for i, word in enumerate(words)]
words = (w for w in words)
# trie = StringTrie(words=words)

# print(trie)
# print('Mani' in trie)
# candidates = trie.search('Manilla', max_distance=3)
# print(candidates)

trie2 = ArrayTrie(words = words, no_array_for_single_value=True)
print(trie2)
print(trie2.search("Hello", "fuzzy", 49))
