from collections import defaultdict
import json
import os
import pandas as pd
from pickle import dump, load
from pympler import asizeof
import random
from rapidfuzz.distance.DamerauLevenshtein import distance
from rtrie import Trie
from rtrie.node import AttributeNode, StringAttributeNode
from rtrie.naive_trie import NaiveTrie
from sortedcontainers import SortedDict, SortedSet
import time

sample_dir = 'data'

SAVE=True

random.seed(42)
pd.set_option('display.float_format', lambda x: '%.8f' % x)

def get_all_prefixes(words, prefix):
    start_index = find_prefix_start(words, prefix)
    if start_index == -1:
        return []  # No prefix found

    # Collect all matching prefixes
    prefixes = []
    i = start_index
    while i < len(words) and words[i].startswith(prefix):
        prefixes.append(words[i])
        i += 1
    
    return prefixes

def get_all_startswith(words, prefix):
    start_index = find_prefix_start(words, prefix)
    if start_index == -1:
        return []  # No prefix found

    # Collect all matching prefixes
    prefixes = []
    i = start_index
    while i < len(words) and prefix.startswith(words[i]):
        prefixes.append(prefix)
        i += 1
    
    return prefixes

def find_prefix_start(words, prefix):
    low, high = 0, len(words) - 1
    while low <= high:
        mid = (low + high) // 2
        # Check if the current word starts with the prefix
        if words[mid].startswith(prefix):
            # If it's not the first element and the previous one also matches, search left
            if mid > 0 and words[mid - 1].startswith(prefix):
                high = mid - 1
            else:
                return mid
        elif words[mid] < prefix:
            low = mid + 1
        else:
            high = mid - 1
    return -1  # If no match is found

def find_near_words(sorted_words, target, max_dist):
    n = len(sorted_words)
    left = 0
    right = n - 1
    
    # Binary search to find the closest index
    while left <= right:
        mid = (left + right) // 2
        if sorted_words[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # Expand outwards from the nearest point found
    closest_words = []
    i, j = right, right + 1
    while i >= 0 or j < n:
        if i >= 0:
            dist = distance(sorted_words[i], target)
            if dist <= max_dist:
                closest_words.append(sorted_words[i])
            i -= 1
        if j < n:
            dist = distance(sorted_words[j], target)
            if dist <= max_dist:
                closest_words.append(sorted_words[j])
            j += 1

    return closest_words

def sample_from_file(file, n, num_lines=None):
    """
    Generates a random sample of n lines from a file
    """
    samples = {}

    if num_lines is None:
        num_lines = sum(1 for line in open(file))

    chance = n / num_lines

    with open(file) as f:
        while True:
            for line in f:
                if random.random() < chance:
                    label, id = line.strip().split('\t')
                    if label not in samples:
                        samples[label] = id
                    if len(samples) == n:
                        return samples
                    
            f.seek(0)

def bench_lexicon(type, words, length):
    results = {}

    if type == 'set':
        structure = set()
    elif type == 'sorted_set':
        structure = SortedSet()
    elif type.startswith('rtrie'):
        structure = Trie(node_type = AttributeNode)
    else:
        raise ValueError(f"Invalid type: {type}")

    if type.startswith('rtrie'):
        start = time.time()
        sorted_words = iter(sorted(words, key=lambda x: x))
        print(f"Time to sort: {time.time() - start}")

        start = time.time()
        structure.add_words(sorted_words)
        assert(length == len(structure))
        results['build'] = (time.time() - start)
        del structure

        # create new empty Trie for insert
        structure = Trie(node_type = AttributeNode)
    
    # insert
    start = time.time()
    for word in words:
        structure.add(word)
    assert(length == len(structure))
    results['insert'] = (time.time() - start)

    if not type.startswith('rtrie'):
        results['build'] = results['insert']

    # size
    results['size'] = asizeof.asizeof(structure)

    # contains
    start = time.time()
    for word in words:
        word in structure
    results['contains'] = (time.time() - start)

    # prefixes of
    if type.startswith('rtrie'):
        start = time.time()
        for word in words:
            structure.prefixes_of(word)
        results['prefixes_of'] = (time.time() - start)
    elif type == 'set':
        start = time.time()
        for word in words:
            [w for w in structure if word.startswith(w)]
        results['prefixes_of'] = (time.time() - start)
    else:
        start = time.time()
        for word in words:
            get_all_prefixes(structure, word)
        results['prefixes_of'] = (time.time() - start)

    # starts with
    if type.startswith('rtrie'):
        start = time.time()
        for word in words:
            structure.starts_with(word)
        results['starts_with'] = (time.time() - start)
    elif type == 'set':
        start = time.time()
        for word in words:
            [w for w in structure if w.startswith(word)]
        results['starts_with'] = (time.time() - start)
    else:
        start = time.time()
        for word in words:
            get_all_startswith(structure, word)
        results['starts_with'] = (time.time() - start)

    # edit distance
    if type.startswith('rtrie'):
        start = time.time()
        for word in words:
            structure.similar_to(word, 'distance', 2)
        results['edit_distance'] = (time.time() - start)
    elif type == 'set':
        start = time.time()
        for word in words:
            [w for w in structure if distance(word, w) <= 2]
        results['edit_distance'] = (time.time() - start)
    else:
        start = time.time()
        for word in words:
            find_near_words(structure, word, 2)
        results['edit_distance'] = (time.time() - start)

    # remove
    start = time.time()
    for word in words:
        structure.remove(word)
    results['delete'] = (time.time() - start)

    del structure

    return results
    
def bench_dict(type, entries, length):
    results = {}
    
    if type == 'dict':
        structure = {}
    elif type == 'sorted_dict':
        structure = SortedDict()
    elif type.startswith('rtrie'):
        structure = Trie(node_type = AttributeNode)
    else:
        raise ValueError(f"Invalid type: {type}")

    if type.startswith('rtrie'):
        start = time.time()
        sorted_entries = iter(sorted(entries, key=lambda x: x[0]))
        print(f"Time to sort: {time.time() - start}")

        start = time.time()
        structure.add_words(sorted_entries)
        results['build'] = (time.time() - start)
        del structure

        # create new empty Trie for insert
        structure = Trie(node_type = AttributeNode)
    
    # insert
    start = time.time()
    for word, attributes in entries.items():
        structure[word] = attributes
    assert(length == len(structure))
    results['insert'] = (time.time() - start)

    if not type.startswith('rtrie'):
        results['build'] = results['insert']

    # size
    results['size'] = asizeof.asizeof(structure)

    # contains
    start = time.time()
    for word in entries.keys():
        word in structure
    results['contains'] = (time.time() - start)

    # prefixes of
    if type.startswith('rtrie'):
        start = time.time()
        for word in entries.keys():
            structure.prefixes_of(word)
        results['prefixes_of'] = (time.time() - start)
    elif type == 'dict':
        start = time.time()
        for word in entries.keys():
            [w for w in structure if w.startswith(word)]
        results['prefixes_of'] = (time.time() - start)
    else:
        start = time.time()
        for word in entries.keys():
            get_all_prefixes(structure.keys(), word)
        results['prefixes_of'] = (time.time() - start)

    # starts with
    if type.startswith('rtrie'):
        start = time.time()
        for word in entries.keys():
            structure.starts_with(word)
        results['starts_with'] = (time.time() - start)
    elif type == 'dict':
        start = time.time()
        for word in entries.keys():
            [w for w in structure if word.startswith(w)]
        results['starts_with'] = (time.time() - start)
    else:
        start = time.time()
        for word in entries.keys():
            get_all_startswith(structure.keys(), word)
        results['starts_with'] = (time.time() - start)

    # edit distance
    if type.startswith('rtrie'):
        start = time.time()
        for word in entries.keys():
            structure.similar_to(word, 'distance', 2)
        results['edit_distance'] = (time.time() - start)
    elif type == 'dict':
        start = time.time()
        for word in entries.keys():
            [w for w in structure if distance(word, w) <= 2]
        results['edit_distance'] = (time.time() - start)
    else:
        start = time.time()
        for word in entries.keys():
            find_near_words(structure.keys(), word, 2)
        results['edit_distance'] = (time.time() - start)

    # remove
    start = time.time()
    for word in entries.keys():
        del structure[word]
    assert(len(structure) == 0)
    results['delete'] = (time.time() - start)

    del structure

    return results

num_iterations = 5
num_runs = 5
sizes = [1, 10, 100, 1000]
totals = defaultdict(list)

for size in sizes:
    # time complexity
    for i in range(num_iterations):
        print(f"Size: {size}, Iteration {i}")
        sample_path = f'{sample_dir}/latest-no-academic-papers-10_000_000.tsv'
        
        samples = sample_from_file(sample_path, size, num_lines=10000000)
        length = len(samples.keys())
        assert(length == size)

        # if size == 100:
        #     with open(f'{sample_dir}/samples_{size}.json', 'w') as f:
        #         print(samples)
        #         json.dump([list(entry) for entry in samples], f)

        if size not in totals:
            totals[size] = defaultdict(list)

        runs = defaultdict(dict)
        
        for structure in ['set', 'sorted_set', 'rtrie_lexicon']:
            print(f"Running {structure} for {size} samples")
            df = pd.DataFrame()
            for j in range(num_runs):
                df = pd.concat([df, pd.DataFrame(bench_lexicon(structure, samples.keys(), length), index=[j])])

            totals[size][structure].append({'mean': df.mean(), 'std': df.std(), 'min': df.min(), 'max': df.max()})

        for structure in ['dict', 'sorted_dict', 'rtrie']:
            print(f"Running {structure} for {size} samples")
            df = pd.DataFrame()
            for j in range(num_runs):
                df = pd.concat([df, pd.DataFrame(bench_dict(structure, samples, length), index=[j])])

            totals[size][structure].append({'mean': df.mean(), 'std': df.std(), 'min': df.min(), 'max': df.max()})

output_dir = 'benchmarks/results'
os.makedirs(output_dir, exist_ok=True)

results = defaultdict(list)
for size in totals.keys():
    for structure in totals[size].keys():
          
        if size not in results:
            results[size] = {}

        dfs = [pd.DataFrame(result) for result in totals[size][structure]]
        totals[size][structure] = pd.concat(dfs, keys=range(num_iterations)).groupby(level=1).mean()
        with open(f'{output_dir}/{structure}_{size}.csv', 'w') as f:
            f.write(pd.DataFrame(totals[size][structure]).to_csv())

