from collections import defaultdict
import os
import pandas as pd
from pickle import dump, load
from pympler import asizeof
import random
from rtrie import Trie
from rtrie.node import AttributeNode, StringAttributeNode
from rtrie.naive_trie import NaiveTrie
from sortedcontainers import SortedDict, SortedSet
import time

sample_dir = 'data'

SAVE=True

random.seed(42)
pd.set_option('display.float_format', lambda x: '%.8f' % x)

def sample_from_file(file, n, condition=lambda x: True, num_lines=None):
    """
    Sample n lines from a file, with a condition to filter lines.
    Works for both sorted and unsorted files.
    """
    samples = set()
    if num_lines is None:
        num_lines = sum(1 for line in open(file))

    with open(file) as f:
        for i, line in enumerate(f):
            if random.random() < n / num_lines and condition(line):
                samples.add(line)
            if len(samples) == n:
                break
            
    return samples
    entries = [tuple(word.strip().split('\t')) for word in samples]
    words = [w[0] for w in entries]

    results = {}

    start = time.time()
    hash_set = set(words)
    data_structures.append('hash_set')
    times.append(time.time() - start)
    sizes.append(asizeof.asizeof(hash_set))
    if SAVE:
      dump(hash_set, open(f'data/hash_set_{size}.pkl', 'wb'))
    del hash_set

    start = time.time()
    hash_map = {word: id for word, id in entries}
    data_structures.append('hash_map')
    times.append(time.time() - start)
    sizes.append(asizeof.asizeof(hash_map))
    if SAVE:
      dump(hash_map, open(f'data/hash_map_{size}.pkl', 'wb'))
    del hash_map

    start = time.time()
    sorted_set = SortedSet(words)
    data_structures.append('sorted_set')
    times.append(time.time() - start)
    sizes.append(asizeof.asizeof(sorted_set))
    if SAVE:
      dump(sorted_set, open(f'data/sorted_set_{size}.pkl', 'wb'))
    del sorted_set

    start = time.time()
    sorted_dict = SortedDict({word: id for word, id in entries})
    data_structures.append('sorted_dict')
    times.append(time.time() - start)
    sizes.append(asizeof.asizeof(sorted_dict))
    if SAVE:
      dump(sorted_dict, open(f'data/sorted_dict_{size}.pkl', 'wb'))
    del sorted_dict

    start = time.time()
    trie = Trie(words = iter(words))
    data_structures.append('trie')
    times.append(time.time() - start)
    sizes.append(asizeof.asizeof(trie))
    if SAVE:
      dump(trie, open(f'data/trie_{size}.pkl', 'wb'))
    del trie

    start = time.time()
    naive_trie = NaiveTrie(words = words)
    data_structures.append('naive_trie')
    times.append(time.time() - start)
    sizes.append(asizeof.asizeof(naive_trie))
    if SAVE:
      dump(naive_trie, open(f'data/naive_trie_{size}.pkl', 'wb'))
    del naive_trie

def bench(type, words, node_type = AttributeNode):
    results = {}

    if type == 'hash_set':
        structure = set()
    elif type == 'sorted_set':
        structure = SortedSet()
    elif type == 'hash_map':
        structure = {}
    elif type == 'sorted_dict':
        structure = SortedDict()
    elif type == 'trie':
        structure = Trie(node_type = node_type)
    elif type == 'naive_trie':
        structure = NaiveTrie()

    # build, only Trie has optimization, the rest just iterate over insert
    if type == 'trie':
        start = time.time()
        sorted_words = iter(sorted(words))
        print(f"Time to sort: {time.time() - start}")

        start = time.time()
        structure.add_words(sorted_words)
        results['build'] = (time.time() - start)
        del structure
        structure = Trie(node_type = node_type)
        
    # insert
    start = time.time()
    for word in words:
        structure.add(word)
    results['insert'] = (time.time() - start)
    
    if type != 'trie':
        results['build'] = results['insert']

    # size
    results['size'] = asizeof.asizeof(structure)

    # contains
    start = time.time()
    for word in words:
        word in structure
    results['contains'] = (time.time() - start)

    # remove
    start = time.time()
    for word in words:
        structure.remove(word)
    results['delete'] = (time.time() - start)
      
    del structure

    return results
    times = {}

    if type == 'hash_map':
        structure = {}
    elif type == 'sorted_dict':
        structure = SortedDict()
    elif type == 'trie':
        structure = Trie()
    elif type == 'naive_trie':
        structure = NaiveTrie()

    # build
    start = time.time()
    for word in words:
        structure[word] = 1
        
    # insert
    start = time.time()
    for word in words:
        structure.add(word)
    times['insert'] = (time.time() - start)

    # size
    size = asizeof.asizeof(structure)

    # contains
    start = time.time()
    for word in words:
        word in structure
    times['contains'] = (time.time() - start)

    # remove
    start = time.time()
    for word in words:
        del structure[word]
    times['delete'] = (time.time() - start)

    # if SAVE:
    #   dump(structure, open(f'data/{structure}_{size}.pkl', 'wb'))
      
    del structure

structures = ['hash_set', 'sorted_set', 'hash_map', 'sorted_dict', 'trie', 'naive_trie']
num_iterations = 2
num_runs = 5
sizes = [1]
totals = defaultdict(list)

for i in range(num_iterations):
    print(f"Iteration {i}")
    for size in sizes:
        samples = [sample.strip().split('\t') for sample in sample_from_file(f'{sample_dir}/latest-no-academic-papers-1.tsv', size, num_lines=61058021)]
        
        if size not in totals:
            totals[size] = defaultdict(list)

        runs = defaultdict(dict)
        for structure in structures:
            print(f"Running {structure} for {size} samples")
            df = pd.DataFrame()
            for j in range(num_runs):
                words = ([w[0] for w in samples])
                entries = (w for w in samples)
                df = pd.concat([df, pd.DataFrame(bench(structure, words), index=[j])])

            totals[size][structure].append({'mean': df.mean(), 'std': df.std(), 'min': df.min(), 'max': df.max()})

output_dir = 'benchmarks/results'
os.makedirs(output_dir, exist_ok=True)

results = defaultdict(list)
for size in totals.keys():
    for structure in totals[size].keys():
          
        if size not in results:
            results[size] = {}

        print(f"Results for {structure} with {size} samples")
        dfs = [pd.DataFrame(result) for result in totals[size][structure]]
        totals[size][structure] = pd.concat(dfs, keys=range(num_iterations)).groupby(level=1).mean()
        with open(f'{output_dir}/{structure}_{size}.csv', 'w') as f:
            f.write(pd.DataFrame(totals[size][structure]).to_csv())



        
