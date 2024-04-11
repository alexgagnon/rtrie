from rtrie import Trie
import logging
import os
from rtrie.node import AttributeNode, MaxLengthStringAttributeNode
from rtrie.array_trie import ArrayTrie
from rtrie.string_trie import StringTrie
from rtrie.naive_trie import NaiveTrie
from sortedcontainers import SortedDict, SortedSet
import sys
from pympler import asizeof

level = logging.DEBUG if os.environ.get("DEBUG") == "True" else logging.INFO
logging.basicConfig(level = level)

# words = ['mac', 'mama', 'mother', 'there']
# trie = Trie(words=iter(words), node_type=AttributeNode)
# # print(trie.prefixes_of('mother'))
# # print(trie.prefixes_of('therein'))
# print(trie.edit_distance('moth', 2))

words = ["Hey", "H", "There", "Hello", "Hi"]
trie = Trie(words = iter(sorted(words)))
print(list(trie.starts_with("There")))

# with open('data/sample.tsv') as f:
    # entries = [tuple(word.strip().split("\t")) for word in f]
    # words = [w[0] for w in entries]
    # words_gen = (w for w in words)
    # entries_gen = (entry for entry in entries)
    # hash_set = set(words)
    # hash_map = {word: id for word, id in entries}
    # sorted_set = SortedSet(words)
    # sorted_map = SortedDict(hash_map)
    # trie = Trie(words=words_gen)
    # string_trie = Trie(words=entries_gen, node_type=MaxLengthStringAttributeNode)
    # naive_trie = NaiveTrie(words=words)

    # # print(hash_set)
    # # print(hash_map)
    # # print(sorted_set)
    # # print(sorted_map)
    # # print(trie)
    # # print(string_trie)
    # # print(naive_trie)

    # print(f'HashSet size: {asizeof.asizeof(hash_set)}')
    # print(f'HashMap size: {asizeof.asizeof(hash_map)}')
    # print(f'SortedSet size: {asizeof.asizeof(sorted_set)}')
    # print(f'SortedMap size: {asizeof.asizeof(sorted_map)}')
    # print(f'Trie size: {asizeof.asizeof(trie)}')
    # print(f'String trie size: {asizeof.asizeof(string_trie)}')
    # print(f'Naive trie size: {asizeof.asizeof(naive_trie)}')

    # # term = 'Belgium'
    # # print(term in hash_set)
    # # print(term in hash_map)
    # # print(term in sorted_set)
    # # print(term in sorted_map)
    # # print(term in trie)
    # # print(term in naive_trie)

    # print(len(string_trie))
    # string_trie.add('dog', 24)
    # print(string_trie)
    # print(string_trie.root.max_length)

    # # _123 = sys.intern('123')
    # # print(asizeof.asizeof(_123))
    # # print(asizeof.asizeof(123))
    # # print(asizeof.asizeof('123'))
    # # print(asizeof.asizeof(_123))
    # # print(asizeof.asizeof('123|456'))
    # # print(asizeof.asizeof([123, 456]))
    # # print(asizeof.asizeof((123, 456)))
    # # print(asizeof.asizeof(['123', '456']))
    # # print(asizeof.asizeof([_123, '456']))
    # # print(asizeof.asizeof(['123', '123']))
    # # print(asizeof.asizeof([_123, _123]))

    # # print(trie.stats())

    # # print('Mani' in trie)
    # # candidates = trie.search('Manilla', max_distance=3)
    # # print(candidates)

    # # trie2 = ArrayTrie(words=words, no_array_for_single_value=True)
    # # print(trie2)
    # # print(trie2.search("Hello", "fuzzy", 49))
    # # print(trie2.search("Hello", "edit", 4))

