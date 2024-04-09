# rtrie

A generalized radix trie implementation in pure python.

## TODO

- [ ] starts_with method
- [ ] running edit distance
- [ ] sys.intern dict keys

## Usage

### Building

The fastest way to build a Trie is by having the words in sorted order and then using the `add_words` method.
Alternatively you can add a collection of words out of order using the `add` method, however due to shuffling nodes this can take substantially longer

### Searching

There are two kinds of search, exact lookup (i.e. `<term> in <trie>`) and candidate search.

Candidate search has two included algorithms: edit distance and similarity ratio

**NOTE: for similarity ratio to work well, you need to use MaxDepthNode instead of Node, since the inclusion of the depth allows for optimizing the search path. MaxDepthNode includes one additional slot, `max_letters`.**

This optimization works by comparing the current similarity ratio to what it could be given the remaining letters in the path, i.e. if the remaining letters were all incorrect and it still passes the threshold, then add the entire subtrie to the list of candidates. Otherwise if we assume they were all correct and it doesn't pass the threshold, then we can skip the entire subtrie.