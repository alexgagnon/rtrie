# rtrie

A generalized radix trie implementation in pure python.

## Usage

### Building

The fastest way to build a Trie is by having the words in sorted order and then using the `add_words` method.
Alternatively you can add a collection of words out of order using the `add` method, however due to shuffling nodes this can take substantially longer

## TODO

- [ ] running edit distance