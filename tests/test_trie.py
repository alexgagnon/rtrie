import pytest
from rtrie import AttributeNode, StringAttributeNode, Trie, get_longest_prefix_index, get_longest_prefixes_index
from deepdiff import DeepDiff
import json
from logging import info
from rapidfuzz.distance.DamerauLevenshtein import distance

# NOTE:
# DeepDiff returns an empty object if the two values are the same, and empty dictionaries
# are Falsey, so we use `not DeepDiff(...)` or `DeepDiff(...) == {}` to check for equality

words = ['Hello', 'Hey', 'I', 'Man', 'Man', 'Manta', 'Manitee', 'There']
add_words = ['Hell', 'M', 'Man', 'Theres']

def test_trie():
    trie = Trie()
    trie.add_words(iter(words))
    for word in words:
        assert(word in trie)
    assert('He' not in trie)
    assert(len(trie) == 7)
    for word in add_words:
        trie.add(word)
    for word in words + add_words:
        assert(word in trie)
    assert('He' not in trie)
    assert(len(trie) == 10)
    assert(trie['Hey'] == ('Hey', True))

    with pytest.raises(KeyError):
        trie['Nope']

def test_add_words_vs_iter_add_lexicon():
    # AttributeNode
    with open('examples/100.tsv') as f:
        entries = [entry.strip().split('\t')[0] for entry in f]
        print(entries)

        trie_1 = Trie(words=iter(sorted(entries)), node_type=AttributeNode)
        trie_2 = Trie(node_type=AttributeNode)
        for entry in entries:
            trie_2.add(entry)
        
        # NOTE: can't use str representation because the order of the keys traversed is not guaranteed
        assert(DeepDiff(trie_1, trie_2) == {})
        assert(len(trie_1) == len(trie_2))
        assert(len(trie_1) == 100)

def test_add_words_vs_iter_add_dictionary():
    # StringAttributeNode
    with open('examples/100.tsv') as f:
        entries = [tuple(entry.strip().split('\t')) for entry in f]

        trie_1 = Trie(words=iter(sorted(entries, key=lambda x: x[0])), node_type=StringAttributeNode)
        trie_2 = Trie(node_type=StringAttributeNode)
        for entry in entries:
            trie_2.add(entry[0], entry[1])
        
        # NOTE: can't use str representation because the order of the keys traversed is not guaranteed
        assert(DeepDiff(trie_1, trie_2) == {})
        assert(len(trie_1) == len(trie_2))
        assert(len(trie_1) == 100)

def test_single_item():
    trie = Trie()
    trie.add("Hello")
    expected = {}
    expected["Hello"] = AttributeNode(True)
    assert(DeepDiff(trie.root.children, expected) == {})

def test_two_unique_items():
    trie = Trie()
    trie.add("Hello")
    trie.add("There")
    expected = {}
    expected["Hello"] = AttributeNode(True)
    expected["There"] = AttributeNode(True)
    assert(DeepDiff(trie.root.children, expected) == {})

def test_word_longer_than_existing_key():
    trie = Trie()
    trie.add("Man")
    trie.add("Manitoba")
    expected = {}
    expected["Man"] = AttributeNode(True, {'itoba': AttributeNode(True)})
    assert(DeepDiff(trie.root.children, expected) == {})

def test_word_exactly_key():
    trie = Trie()
    trie.add("Man")
    trie.add("Man")
    expected = {}
    expected["Man"] = AttributeNode(True)
    assert(DeepDiff(trie.root.children, expected) == {})

def test_shared_prefix():
    trie = Trie()
    trie.add("Manta")
    trie.add("Manitoba")
    expected = {}
    expected["Man"] = AttributeNode(None, {'ta': AttributeNode(True), 'itoba': AttributeNode(True)})
    assert(DeepDiff(trie.root.children, expected) == {})

def test_shared_prefix_exact():
    trie = Trie()
    trie.add("Manta")
    trie.add("Manitoba")
    trie.add("Man")
    expected = {}
    expected["Man"] = AttributeNode(True, {'ta': AttributeNode(True), 'itoba': AttributeNode(True)})
    assert(DeepDiff(trie.root.children, expected) == {})

def test_contains():
    trie = Trie()
    assert("Manta" not in trie)

    trie.add("Manta")
    assert("Manta" in trie)
    assert("Mantas" not in trie)

    trie.add("Manitoba")
    assert("Manta" in trie)
    assert("Man" not in trie)
    assert("Manitoba" in trie)

    words = ["Man♥ta", "Man♥itoba", "Maybe♥", "Man", "Hello", "There", "Manitee", "Marquee", "Mobile", "Marble"]
    trie = Trie(words = iter(sorted(words)))
    for word in words:
        assert(word in trie)

def test_match():
    assert(get_longest_prefix_index("one", "two") == 0)
    assert(get_longest_prefix_index("one", "other") == 1)
    assert(get_longest_prefix_index("one", "one") == 3)
    assert(get_longest_prefix_index("one", "oneid") == 3)

def test_words():
    words = ["Hey", "There", "Hello", "Hi"]
    trie = Trie(words = iter(sorted(words)))

    assert(DeepDiff(words, list(trie)))
    assert(sorted(words) == list(trie.words(sort = True)))

def test_nodes():
    words = ["Hey", "There", "Hello", "Hi"]
    trie = Trie(words = iter(sorted(words)))
    num_nodes = sum(1 for _ in trie.nodes())
    assert(num_nodes == 6)

def test_delete():
    words = ["Help", "He", "There", "Hello", "Hi", "Hel"]
    trie = Trie(words = iter(sorted(words)))
    info(trie)
    assert(len(trie) == 6)

    # should do nothing, 'H' is an intermediate node that isn't a word
    del trie["H"]
    assert('H' not in trie)
    assert(len(trie) == 6)

    info("Deleted 'H'")
    info(trie)

    # should reduce number of words but not change structure, since
    # 'Hel' is an intermediate node that is a word, with multiple children
    del trie["Hel"]
    assert('Hel' not in trie)
    assert(len(trie) == 5)

    info("Deleted 'Hel'")
    info(trie)

    # should delete leaf
    assert('Help' in trie)
    del trie["Help"]
    assert('Help' not in trie)
    assert(len(trie) == 4)

    info("Deleted 'Help'")
    info(trie)

    # should delete leaf node who's parent is root
    assert('There' in trie)
    del trie["There"]
    assert('There' not in trie)
    assert(len(trie) == 3)

    info("Deleted 'There'")
    info(trie)

    del trie['Hi']
    assert('Hi' not in trie)
    assert(len(trie) == 2)

    info("Deleted 'Hi'")
    info(trie)

    del trie['Hello']
    assert('Hello' not in trie)
    assert(len(trie) == 1)

    info("Deleted 'Hello'")
    info(trie)

    del trie['He']
    assert('He' not in trie)
    assert(len(trie) == 0)

    info("Deleted 'He'")
    info(trie)

def test_starts_with():
    words = ["Hey", "There", "Hello", "Hi"]
    trie = Trie(words = iter(sorted(words)))
    assert(not DeepDiff(list(trie.starts_with("H")), [("Hey", True), ("Hello", True), ("Hi", True)], ignore_order=True))
    assert(not DeepDiff(list(trie.starts_with("He")), [("Hey", True), ("Hello", True)], ignore_order=True))
    assert(not DeepDiff(list(trie.starts_with("There")), [("There", True)], ignore_order=True))
    assert(not DeepDiff(list(trie.starts_with("Th")), [("There", True)], ignore_order=True))
    assert(not DeepDiff(list(trie.starts_with("T")), [("There", True)], ignore_order=True))
    assert(not DeepDiff(list(trie.starts_with("")), [("Hey", True), ("There", True), ("Hello", True), ("Hi", True)], ignore_order=True))
    
def test_prefixes_of():
    words = ["Hey", "H", "There", "Hello", "Hi"]
    trie = Trie(words = iter(sorted(words)))
    assert(trie.prefixes_of("Heyllo") == [("H", True),("Hey", True)])
    assert(trie.prefixes_of("Therein") == [("There", True)])
    assert(trie.prefixes_of("H") == [("H", True)])
    assert(trie.prefixes_of("Hii") == [("H", True), ("Hi", True)])
    assert(trie.prefixes_of("Hiiii") == [("H", True), ("Hi", True)])
    assert(trie.prefixes_of("L") == [])

def test_edit_distance():
    words = ["Hey", "H", "There", "Hello", "Hi"]
    trie = Trie(words = iter(sorted(words)))
    hash_set = set(words)
    for word in words:
        assert(trie.edit_distance(word, 0) == [(0, word, True)])
        assert([(0, w, True) for w in hash_set if distance(word, w) == 0] == [(0, word, True)])
    assert(trie.edit_distance("Magna", 4) == [])
    assert(not DeepDiff(trie.edit_distance("H", 1), [(0, "H", True), (1, "Hi", True)], ignore_order=True))
    assert(not DeepDiff(trie.edit_distance("z", 1), [(1, "H", True)], ignore_order=True))
    assert(not DeepDiff(trie.edit_distance("H", 2), [(0, "H", True), (1, "Hi", True), (2, "Hey", True)], ignore_order=True))
    assert(not DeepDiff(trie.edit_distance("H", 3), [(0, "H", True), (1, "Hi", True), (2, "Hey", True)], ignore_order=True))
    assert(not DeepDiff(trie.edit_distance("ey", 1), [(1, "Hey", True)], ignore_order=True))
    assert(not DeepDiff(trie.edit_distance("ey", 2), [(2, "H", True), (2, "Hi", True), (1, "Hey", True)], ignore_order=True))

def _get_hash_distance(words, word, max_distance):
    return [(distance(word, w), w, True) for w in words if distance(word, w) <= max_distance]

def test_get_longest_prefixes_index():
    words = ["Hello", "Hi"]
    assert(get_longest_prefixes_index(words) == 1)
    words = ["Hello", "Hey"]
    assert(get_longest_prefixes_index(words) == 2)
    words = ["Hello", "Heli", "Hey"]
    assert(get_longest_prefixes_index(words) == 2)
    words = ["Hello", "Hi", "Hola"]
    assert(get_longest_prefixes_index(words) == 1)
    words = ["Hello", "There"]
    assert(get_longest_prefixes_index(words) == 0)

def test_stats():
    words = ['Hello', 'Hey', 'Man', 'Man', 'Manitee', 'There']
    trie = Trie(words=iter(words))
    stats = trie.stats(unique=True)
    expect = {
        'num_words': 5,
        'average_length': 4.6,
        'word_lengths': {5: 2, 3: 2, 7: 1},
        'letter_frequency': {'H': 2, 'e': 6, 'l': 2, 'o': 1, 'y': 1, 'M': 2, 'a': 2, 'n': 2, 'i': 1, 't': 1, 'T': 1, 'h': 1, 'r': 1},
        'letter_distribution': {
            'H': {
                0: 2,
            }, 
            'e': {
                1: 2,
                2: 1,
                4: 1,
                5: 1,
                6: 1,
            }, 
            'l': {
                2: 1,
                3: 1,
            }, 
            'o': {
                4: 1
            },
            'y': {
                2: 1
            },
            'M': {
                0: 2,
            }, 
            'a': {
                1: 2
            }, 
            'n': {
                2: 2
            }, 
            'i': {
                3: 1
            }, 
            't': {
                4: 1
            }, 
            'T': {
                0: 1
            }, 
            'h': {
                1: 1
            }, 
            'r': {
                3: 1
            }
        },
        'num_nodes': 6,
        'node_distribution': {1: 3, 2: 3},
        'lengths_at_node_depths': {1: {2: 1, 5: 1, 3: 1}, 2: {5: 1, 3: 1, 7: 1}}
    }

    assert(DeepDiff(expect, stats) == {})