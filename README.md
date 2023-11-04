# rtrie

A generalized radix trie implementation in pure python.

## TODO

- [ ] investigate microdict

## Usage

### Building

#### Saving subtries to disk

When dealing with data that when stored would exceed your available memory, you can define the max depth you want to maintain in memory, and any data that exceeds that depth will be saved to disk. This is done by passing a `save_path` to the `build` method. It will automatically reload the trie from disk when needed.

### Searching

There are two kinds of search, exact lookup and candidate search.

Candidate search has two included algorithms: edit distance and similarity ratio

**NOTE: for similarity ratio to work well, you need to use MaxDepthNode instead of Node, since the inclusion of the depth allows for optimizing the search path. MaxDepthNode includes one addition field, `max_letters`, which is negligible compared to the size of the trie.**

This optimization works by comparing the current similarity ratio to what it could be given the remaining letters in the path, i.e. if the remaining letters were all incorrect and it still passes the threshold, then add the entire subtrie to the list of candidates. Otherwise if we assume they were all correct and it doesn't pass the threshold, then we can skip the entire subtrie.