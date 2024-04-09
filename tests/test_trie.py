import pytest
from rtrie import AttributeNode, Trie, get_longest_prefix_index, get_longest_prefixes_index
from deepdiff import DeepDiff
        
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
    assert(trie['Nope'] == None)
    
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
    print(trie)
    assert(len(trie) == 6)

    # should do nothing, 'H' is an intermediate node that isn't a word
    del trie["H"]
    assert('H' not in trie)
    assert(len(trie) == 6)

    print("Deleted 'H'")
    print(trie)

    # should reduce number of words but not change structure, since
    # 'Hel' is an intermediate node that is a word, with multiple children
    del trie["Hel"]
    assert('Hel' not in trie)
    assert(len(trie) == 5)

    print("Deleted 'Hel'")
    print(trie)

    # should delete leaf
    assert('Help' in trie)
    del trie["Help"]
    assert('Help' not in trie)
    assert(len(trie) == 4)

    print("Deleted 'Help'")
    print(trie)

    # should delete leaf node who's parent is root
    assert('There' in trie)
    del trie["There"]
    assert('There' not in trie)
    assert(len(trie) == 3)

    print("Deleted 'There'")
    print(trie)

    del trie['Hi']
    assert('Hi' not in trie)
    assert(len(trie) == 2)

    print("Deleted 'Hi'")
    print(trie)

    del trie['Hello']
    assert('Hello' not in trie)
    assert(len(trie) == 1)

    print("Deleted 'Hello'")
    print(trie)

    del trie['He']
    assert('He' not in trie)
    assert(len(trie) == 0)

    print("Deleted 'He'")
    print(trie)


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