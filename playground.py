from rtrie import Trie
import json
import logging
import os
from rtrie.node import AttributeNode, StringAttributeNode, MaxLengthStringAttributeNode
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

# words = ["Hey", "H", "There", "Hello", "Hi"]
# words = ([word, i] for i, word in enumerate(sorted(words)))
# trie = Trie(words=words, node_type=StringAttributeNode)

# with open('data/sample-100.tsv') as f:
    # entries = [tuple(word.strip().split("\t")) for word in f]
with open('data/samples_100.json') as f:
    entries = json.load(f)
    entries = [tuple(entry) for entry in entries]
    print(entries[0])
    words = [w[0] for w in entries]
    words_gen = (w for w in words)
    entries_gen = (entry for entry in entries)
    sorted_words_gen = (w for w in sorted(words))
    sorted_entries_gen = (entry for entry in sorted(entries, key=lambda x: x[0]))
    hash_set = set(words)
    hash_map = {word: id for word, id in entries}
    sorted_set = SortedSet(words)
    sorted_map = SortedDict(hash_map)
    trie = Trie(words=words_gen)
    # string_trie = Trie(words=sorted_entries_gen, node_type=AttributeNode)
    string_trie = Trie(node_type=AttributeNode)
    for entry in sorted(entries, key=lambda x: x[0]):
        string_trie.add(entry[0])
    naive_trie = NaiveTrie(words=words)

    print(string_trie)
    print(len(string_trie))

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


