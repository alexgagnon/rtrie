# rtrie

A generalized radix trie implementation in pure python.

## TODO

- [ ] prefix method
- [ ] running edit distance
- [ ] sys.intern dict keys
- [ ] confirm if nodes/items in Node instead of Trie has impact

## Usage

### Building

The fastest way to build a Trie is by having the words in sorted order and then using the `add_words` method.
Alternatively you can add a collection of words out of order using the `add` method, however due to shuffling nodes this can take substantially longer

### Searching

There are two kinds of search, exact lookup (i.e. `<term> in <trie>`) and candidate search.

Candidate search has two included algorithms: edit distance and similarity ratio

**NOTE: to speed up search, you need to use a subclass of MaxDepthNode, since the inclusion of the depth allows for optimizing the search path. MaxDepthNode includes one additional slot, `max_depth`.**

This optimization works by comparing the current similarity ratio to what it could be given the remaining letters in the path, i.e. if the remaining letters were all incorrect and it still passes the threshold, then add the entire subtrie to the list of candidates. Otherwise if we assume they were all correct and it doesn't pass the threshold, then we can skip the entire subtrie.