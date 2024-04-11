import os
from pympler import asizeof
from rtrie import Trie
from rtrie.node import StringAttributeNode
from rtrie.naive_trie import NaiveTrie
from sortedcontainers import SortedDict, SortedSet
import time

sample_dir = 'data'

for filename in os.listdir(sample_dir):
    if not filename.endswith('.tsv'):
        continue
    
    with open(f'{sample_dir}/{filename}') as f:
        print(f'Processing {filename}')
        entries = [tuple(word.strip().split('\t')) for word in f]
        words = [w[0] for w in entries]
        words_gen = (w for w in words)
        entries_gen = (entry for entry in entries)

        start = time.time()
        hash_set = set(words)
        end = time.time()
        print(f'HashSet time: {end - start}')
        print(f'HashSet size: {asizeof.asizeof(hash_set)}')

        start = time.time()
        hash_map = {word: id for word, id in entries}
        end = time.time()
        print(f'HashMap time: {end - start}')
        print(f'HashMap size: {asizeof.asizeof(hash_map)}')

        start = time.time()
        sorted_set = SortedSet(words)
        end = time.time()
        print(f'SortedSet time: {end - start}')
        print(f'SortedSet size: {asizeof.asizeof(sorted_set)}')

        start = time.time()
        sorted_map = SortedDict(hash_map)
        end = time.time()
        print(f'SortedMap time: {end - start}')
        print(f'SortedMap size: {asizeof.asizeof(sorted_map)}')

        start = time.time()
        trie = Trie(words=words_gen)
        end = time.time()
        print(f'Trie time: {end - start}')
        print(f'Trie size: {asizeof.asizeof(trie)}')

        start = time.time()
        string_trie = Trie(words=entries_gen, node_type=StringAttributeNode)
        end = time.time()
        print(f'String trie time: {end - start}')
        print(f'String trie size: {asizeof.asizeof(string_trie)}')

        start = time.time()
        naive_trie = NaiveTrie(words=words)
        end = time.time()
        print(f'Naive trie time: {end - start}')
        print(f'Naive trie size: {asizeof.asizeof(naive_trie)}')

