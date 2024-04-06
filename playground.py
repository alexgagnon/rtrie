from rtrie import Trie
import logging
import os
from rtrie.array_trie import ArrayTrie
from rtrie.string_trie import StringTrie
from rtrie.naive_trie import NaiveTrie
from sortedcontainers import SortedDict, SortedSet

level = logging.DEBUG if os.environ.get("DEBUG") == "True" else logging.INFO
logging.basicConfig(level = level)

with open('data/sample.tsv') as f:
    entries = [tuple(word.strip().split("\t")) for word in f]
    words = [w[0] for w in entries]
    words_gen = (w for w in words)
    # words = ['mama', 'manta']
    # # words_list = [(word, i) for i, word in enumerate(words)]
    # words_gen = (w for w in words)
    hash_set = set(words)
    sorted_set = SortedSet(words)
    trie = Trie(words=words_gen)
    naive_trie = NaiveTrie(words=words)

    print(hash_set)
    print(sorted_set)
    print(trie)
    print(naive_trie)

    import sys
    from pympler import asizeof

    print(f'HashSet size: {asizeof.asizeof(hash_set)}')
    print(f'SortedSet size: {asizeof.asizeof(sorted_set)}')
    print(f'Trie size: {asizeof.asizeof(trie)}')
    print(f'Naive trie size: {asizeof.asizeof(naive_trie)}')
    # _123 = sys.intern('123')
    # print(asizeof.asizeof(_123))
    # print(asizeof.asizeof(123))
    # print(asizeof.asizeof('123'))
    # print(asizeof.asizeof(_123))
    # print(asizeof.asizeof('123|456'))
    # print(asizeof.asizeof([123, 456]))
    # print(asizeof.asizeof((123, 456)))
    # print(asizeof.asizeof(['123', '456']))
    # print(asizeof.asizeof([_123, '456']))
    # print(asizeof.asizeof(['123', '123']))
    # print(asizeof.asizeof([_123, _123]))

    # print(trie.stats())

    # print('Mani' in trie)
    # candidates = trie.search('Manilla', max_distance=3)
    # print(candidates)

    # trie2 = ArrayTrie(words=words, no_array_for_single_value=True)
    # print(trie2)
    # print(trie2.search("Hello", "fuzzy", 49))
    # print(trie2.search("Hello", "edit", 4))

