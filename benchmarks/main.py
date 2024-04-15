from collections import defaultdict
import json
import os
import sys
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
from objsize import get_deep_size

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

def bench_lexicon(type, words, length, test_words):
    results = {}

    if type == 'set':
        structure = set()
    elif type == 'sorted_set':
        structure = SortedSet()
    elif type.startswith('rtrie'):
        structure = Trie(node_type = AttributeNode)
    else:
        raise ValueError(f"Invalid type: {type}")

    # build/insert
    if type.startswith('rtrie'):
        start = time.time()
        sorted_words = iter(sorted(words, key=lambda x: x))
        print(f"Time to sort: {time.time() - start}")

        start = time.time()
        structure.add_words(sorted_words)
        results['insert'] = (time.time() - start)
        assert(length == len(structure))
    else:
        start = time.time()
        for word in words:
            structure.add(word)
        results['insert'] = (time.time() - start)
        assert(length == len(structure))

    # size
    results['size'] = asizeof.asizeof(structure)

    # contains
    start = time.time()
    for word in test_words:
        word in structure
    results['contains'] = (time.time() - start)

    # prefixes of
    if type.startswith('rtrie'):
        start = time.time()
        for word in test_words:
            structure.prefixes_of(word)
        results['prefixes_of'] = (time.time() - start)
    elif type == 'set':
        start = time.time()
        for word in test_words:
            [w for w in structure if word.startswith(w)]
        results['prefixes_of'] = (time.time() - start)
    else:
        start = time.time()
        for word in test_words:
            get_all_prefixes(structure, word)
        results['prefixes_of'] = (time.time() - start)

    # starts with
    if type.startswith('rtrie'):
        start = time.time()
        for word in test_words:
            structure.starts_with(word)
        results['starts_with'] = (time.time() - start)
    elif type == 'set':
        start = time.time()
        for word in test_words:
            [w for w in structure if w.startswith(word)]
        results['starts_with'] = (time.time() - start)
    else:
        start = time.time()
        for word in test_words:
            get_all_startswith(structure, word)
        results['starts_with'] = (time.time() - start)

    # edit distance
    if type.startswith('rtrie'):
        start = time.time()
        for word in test_words:
            structure.similar_to(word, 'distance', 2)
        results['edit_distance'] = (time.time() - start)
    elif type == 'set':
        start = time.time()
        for word in test_words:
            [w for w in structure if distance(word, w) <= 2]
        results['edit_distance'] = (time.time() - start)
    else:
        start = time.time()
        for word in test_words:
            find_near_words(structure, word, 2)
        results['edit_distance'] = (time.time() - start)

    # remove
    start = time.time()
    for word in test_words:
        structure.remove(word)
    results['delete'] = (time.time() - start)
    assert(len(structure) == length - len(test_words))

    del structure

    return results
    
def bench_dict(type, entries, length, test_words):
    results = {}
    
    if type == 'dict':
        structure = {}
    elif type == 'sorted_dict':
        structure = SortedDict()
    elif type.startswith('rtrie'):
        structure = Trie(node_type = AttributeNode)
    else:
        raise ValueError(f"Invalid type: {type}")

    # build/insert
    if type.startswith('rtrie'):
        start = time.time()
        sorted_entries = iter(sorted(entries, key=lambda x: x[0]))
        print(f"Time to sort: {time.time() - start}")

        start = time.time()
        structure.add_words(sorted_entries)
        results['insert'] = (time.time() - start)
        # assert(length == len(structure))
    else:
        start = time.time()
        for word, attributes in entries.items():
            structure[word] = attributes
        results['insert'] = (time.time() - start)
        assert(length == len(structure))

    # size
    results['size'] = asizeof.asizeof(structure)

    # contains
    start = time.time()
    for word in test_words:
        word in structure
    results['contains'] = (time.time() - start)

    # prefixes of
    if type.startswith('rtrie'):
        start = time.time()
        for word in test_words:
            structure.prefixes_of(word)
        results['prefixes_of'] = (time.time() - start)
    elif type == 'dict':
        start = time.time()
        for word in test_words:
            [w for w in structure if w.startswith(word)]
        results['prefixes_of'] = (time.time() - start)
    else:
        start = time.time()
        for word in test_words:
            get_all_prefixes(structure.keys(), word)
        results['prefixes_of'] = (time.time() - start)

    # starts with
    if type.startswith('rtrie'):
        start = time.time()
        for word in test_words:
            structure.starts_with(word)
        results['starts_with'] = (time.time() - start)
    elif type == 'dict':
        start = time.time()
        for word in test_words:
            [w for w in structure if word.startswith(w)]
        results['starts_with'] = (time.time() - start)
    else:
        start = time.time()
        for word in test_words:
            get_all_startswith(structure.keys(), word)
        results['starts_with'] = (time.time() - start)

    # edit distance
    if type.startswith('rtrie'):
        start = time.time()
        for word in test_words:
            structure.similar_to(word, 'distance', 2)
        results['edit_distance'] = (time.time() - start)
    elif type == 'dict':
        start = time.time()
        for word in test_words:
            [w for w in structure if distance(word, w) <= 2]
        results['edit_distance'] = (time.time() - start)
    else:
        start = time.time()
        for word in test_words:
            find_near_words(structure.keys(), word, 2)
        results['edit_distance'] = (time.time() - start)

    # remove
    start = time.time()
    for word in test_words:
        del structure[word]
    results['delete'] = (time.time() - start)
    print(length, len(structure), len(test_words))
    # assert(len(structure) == length - len(test_words))

    del structure

    return results


output_dir = 'benchmarks/results'
os.makedirs(output_dir, exist_ok=True)

def get_stats(words):
    stats = {}
    stats['max'] = max(len(word) for word in words)
    stats['min'] = min(len(word) for word in words)
    stats['mean'] = sum(len(word) for word in words) / len(words)
    stats['num_non_ascii'] = sum(1 for word in words if not word.isascii())
    return stats

def t():
    num_iterations = 5
    num_runs = 5
    sizes = [10, 100, 1000, 10000, 100000, 1000000]

    for size in sizes:
        totals = defaultdict(list)
        output = {}
        for i in range(num_iterations):
            print(f"Size: {size}, Iteration {i}")
            sample_path = f'{sample_dir}/latest-no-academic-papers-10_000_000.tsv'
            
            samples = sample_from_file(sample_path, size, num_lines=10000000)
            length = len(samples.keys())
            assert(length == size)

            output['stats'] = get_stats(samples.keys())
            test_words = random.sample(sorted(samples.keys()), 5)
            output['test_words'] = test_words

            for structure in ['set', 'sorted_set', 'rtrie_lexicon']:
                print(f"Running {structure} for {size} samples")
                df = pd.DataFrame()
                for j in range(num_runs):
                    df = pd.concat([df, pd.DataFrame(bench_lexicon(structure, samples.keys(), length, test_words), index=[j])])

                totals[structure].append({'mean': df.mean(), 'std': df.std(), 'min': df.min(), 'max': df.max()})

            for structure in ['dict', 'sorted_dict', 'rtrie']:
                print(f"Running {structure} for {size} samples")
                df = pd.DataFrame()
                for j in range(num_runs):
                    df = pd.concat([df, pd.DataFrame(bench_dict(structure, samples, length, test_words), index=[j])])

                totals[structure].append({'mean': df.mean(), 'std': df.std(), 'min': df.min(), 'max': df.max()})

        for structure in totals.keys():
            dfs = [pd.DataFrame(t) for t in totals[structure]]
            totals[structure] = pd.concat(dfs, keys=range(num_iterations)).groupby(level=1).mean().to_dict()

        output['results'] = totals
        with open(f'{output_dir}/results-{size}.json', 'w') as f:
            f.write(json.dumps(output, indent=2))

def total_size(obj, seen=None):
    """Calculate total memory usage of an object in an iterative way, handling __slots__."""
    if seen is None:
        seen = set()  # To track already visited objects and avoid double counting

    # Stack to hold the objects to be processed
    stack = [obj]
    total = 0

    while stack:
        obj = stack.pop()
        obj_id = id(obj)

        print(obj_id)

        if obj_id in seen:
            continue

        seen.add(obj_id)

        total += sys.getsizeof(obj)

        print(total)
        print(obj)
        print(obj.__class__.__name__)

        # Explore __slots__ if available
        if hasattr(obj, '__slots__'):
            print('has slots')
            print(getattr(obj, '__slots__'))
            for slot in getattr(obj, '__slots__', []):
                try:
                    print(slot)
                    attr = getattr(obj, slot)
                    if isinstance(attr, (dict, list, set, tuple, frozenset)):
                        stack.extend(attr)
                    else:
                        stack.append(attr)
                except AttributeError:
                    # Attribute in __slots__ may not be initialized yet
                    continue
                
        # Also, explore __dict__ if available
        elif hasattr(obj, '__dict__'):
            print('has dict')
            total += sys.getsizeof(obj.__dict__)
            for key, value in obj.__dict__.items():
                stack.append(value)

    return total

def space():
    sizes = [1, 10]
    type = 'slots'
    trie_sizes_pympler = []
    trie_sizes_get_deep_size = []
    # trie_sizes_custom_getsizeof = []
    naive_trie_sizes_pympler = []
    naive_trie_sizes_get_deep_size = []
    # naive_trie_sizes_custom_getsizeof = []

    output = defaultdict(dict)
    for size in sizes:
        print(f"Size: {size}")
        samples = sample_from_file(f'{sample_dir}/latest-no-academic-papers-10_000_000.tsv', size, num_lines=10000000)
        length = len(samples.keys())
        assert(length == size)
        output[size]['stats'] = get_stats(samples.keys())

        trie = Trie(words = (word for word in sorted(samples.keys())))
        assert(len(trie) == size)
        trie_sizes_pympler.append(asizeof.asizeof(trie))
        trie_sizes_get_deep_size.append(get_deep_size(trie))
        # trie_sizes_custom_getsizeof.append(total_size(trie))

        del trie

        naive_trie = NaiveTrie(words = samples.keys())
        assert(len(naive_trie) == size)
        naive_trie_sizes_pympler.append(asizeof.asizeof(naive_trie))
        naive_trie_sizes_get_deep_size.append(get_deep_size(naive_trie))
        # naive_trie_sizes_custom_getsizeof.append(total_size(naive_trie))

        del naive_trie
        
    df = pd.DataFrame({
        'rtrie-pympler': trie_sizes_pympler, 
        'rtrie-objsize': trie_sizes_get_deep_size, 
        # 'rtrie-custom': trie_sizes_custom_getsizeof,
        'trie': naive_trie_sizes_pympler,
        'trie-objsize': naive_trie_sizes_get_deep_size,
        # 'trie-custom': naive_trie_sizes_custom_getsizeof
    }, index=sizes)
    output['results'] = df.to_dict()
    with open(f'benchmarks/results/space-{type}.json', 'w') as f:
        f.write(json.dumps(output, indent=2))


# space()
t()