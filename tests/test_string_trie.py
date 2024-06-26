from rtrie import StringTrie
        
words = ['Hello', 'Hey', 'Man', 'Man', 'Mani', 'Manilla', 'Manitee', 'There']
words = [(word, i) for i, word in enumerate(words)]

def test_string_trie():
    trie = StringTrie(words = iter(words))
    for word, i in words:
        assert(word in trie)