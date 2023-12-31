import logging
import os
import pickle
import gc
from itertools import chain, tee
from collections import deque
from collections.abc import MutableMapping
from array import array
from typing import Any, Callable, ItemsView, Iterator, Literal, Optional, TypeAlias, cast
from rapidfuzz.fuzz import ratio
from rapidfuzz.distance.DamerauLevenshtein import distance

Word: TypeAlias = str | tuple[str, Any]
Words: TypeAlias = Iterator[Word]
Attributes: TypeAlias = Any
Entry: TypeAlias = tuple[str, 'AttributeNode']
Record: TypeAlias = tuple[str, Attributes]
Children: TypeAlias = dict[str, 'AttributeNode']
Candidates: TypeAlias = list[tuple[int, str, 'AttributeNode']]
AdditionalAttributes: TypeAlias = Optional[dict[str, Any]]

def get_filename(string: str) -> str:
    return "".join([x if x.isalnum() else "_" for x in string])

def get_longest_prefix_index(word1: str, word2: str):
    """
        Returns the length of the longest prefix between two words
    """
    if word1 == word2:
        return len(word1)
    max = min(len(word1), len(word2))
    for i in range(max):
        if word1[i] != word2[i]:
            return i
    return max


def get_longest_prefixes_index(words: list[str]):
    """
        Returns the index of the longest prefix in a sorted list of words
    """
    if len(words) == 0:
        return 0
    if len(words) == 1:
        return len(words[0])

    # find the first word with a different first character
    index = 0
    char = words[0][0]
    while index < len(words):
        if words[index][0] != char:
            break
        index += 1

    # if there is no word with even a single matching prefix, return 0
    if index - 1 == 0:
        return 0

    return get_longest_prefix_index(words[0], words[index - 1])


def _get_values(element: Word) -> Record:
    """
      Get the value stored in a Node
    """
    try:
        return (element[0], element[1]) if isinstance(element, tuple) else (element, True)
    except IndexError:
        print(element[0])
        return ("null", -1)

class Node:
    """
      Base Node type
    """

class AttributeNode(Node):
    # avoid storing class attributes in '__dict__' to save memory
    __slots__ = ('attributes', 'children')

    attributes: Attributes
    children: Children
    __extras__: AdditionalAttributes

    def __init__(self, attributes: Attributes = None, children: Optional[Children] = None, *args, **kwargs):
        self.attributes = attributes
        self.children = children

    def __str__(self):
        return self

    def print(self, depth: int):
        if self.children == None:
            return ""
        offset = "\t" * depth
        string = ""
        for item in self.children.items():
            key: str = item[0]
            child: AttributeNode = item[1]
            string += "\n" + offset + "\t" + \
                f"{key}({child.attributes}): {child.print(depth+1)}"
        return string

class StoredNode(Node):
    __slots__ = ('filename')

    filename: str

    def __init__(self, attributes: Attributes = None, children: Optional[Children] = None, filename: str = None, *args, **kwargs):
        super().__init__(attributes, children, *args, **kwargs)
        self.filename = filename

class Trie(MutableMapping[str, Attributes]):
    def __init__(self, 
                 root: AttributeNode | None=None, 
                 words: Optional[Words] = None,
                ):
        """
            Initialize a Trie.

            NOTE: `words` must be either a list of strings or a list of tuples of the form (word: str, attributes: Any). If `words` is a list of
            strings, the `attributes` property will be set to `True` for each word.

            You can pass in a custom add function that is used to control how attributes are defined. This can be useful if you need
            special cases for when words conflict or requiring merging of attributes, i.e. if `attributes` is an object and you add
            a word that already exists with other attributes, you may want to overwrite, merge, etc. The default is to simply assign
            the value `True`, indicating it is a word, and adding the same word again will not affect the trie.
        """

        self.root = root if root != None else AttributeNode()
        self.num_words: int = 0
        # self.depth_to_store = depth_to_store
        # self.subtrie_path = subtrie_path
        # if self.depth_to_store != None:
        #     os.mkdir(self.subtrie_path)
        if words:
            self.add_words(words)

    def __delitem__(self, word: str):
        return self.delete(word)

    def __getitem__(self, word: str) -> Record | None:
        result = self._get_node(word)
        if result != None and result[0][1].attributes != None:
            return (result[0][0], result[0][1].attributes)
        return None

    def __setitem__(self, word: str, attributes: Attributes) -> None:
        result = self._get_node(word)
        if result != None:
            result[0][1].attributes = attributes

    def __str__(self):
        return self.root.print(0)

    def __len__(self):
        return self.num_words

    def __contains__(self, word: object) -> bool:
        result = self._get_node(cast(str, word))
        return result != None and result[0][1].attributes != None

    def __iter__(self):
        return self.words()
    
    def post_add_node(self, **kwargs):
        pass

    def add_attributes(self, node: AttributeNode, value: Attributes) -> int:
        """
          The default add method to use when one isn't provided. It uses True/None to indicate whether a node is a word or not
        """
        # if this is a new word, increment the number of words
        # otherwise we are just overwriting attributes which isn't a new word
        is_new = 1
        if node.attributes != None:
            is_new = 0
        node.attributes = value
        return is_new

    def delete_attributes(self, node: AttributeNode) -> int:
        node.attributes = None
        return -1

    def count_attributes(self, node: AttributeNode) -> int:
        return 1

    def add(self, word: str, attributes: Attributes=True):
        """
          Adds a single word to an already constructed Trie. You should use `add_words` to initialize a Trie for performance reasons
        """
        current: AttributeNode = self.root

        while True:
            logging.debug(f'Adding "{word}"')

            if current.children == None:
                logging.debug("No children, initializing")
                current.children = cast(Children, {})

            logging.debug(str(current.children.keys()))

            if len(current.children.keys()) == 0:
                logging.debug(
                    f'Empty children, adding "{word}" with `attributes = {attributes}')
                current.children[word] = AttributeNode()
                self.num_words += self.add_attributes(current.children[word], attributes)
                break

            elif word in current.children.keys():
                logging.debug(
                    f'"{word}" already exists, adding attributes {attributes}')
                self.num_words += self.add_attributes(current.children[word], attributes)
                break

            else:
                match_found = False
                logging.debug("No exact match, checking remaining keys...")
                for key in list(current.children.keys()):
                    index: int = get_longest_prefix_index(key, word)
                    logging.debug(f"Prefix location: {index}")

                    if index == 0:
                        logging.debug(
                            f'"{key}" has no overlapping prefix, continuing...')
                        continue

                    else:
                        match_found = True
                        prefix = word[:index]
                        word_suffix = word[index:]
                        key_suffix = key[index:]

                        logging.debug(
                            f"\nPrefix: {prefix}\nWord remainder: {word_suffix}\nKey remainder: {key_suffix}")

                        if len(key_suffix) == 0:
                            logging.debug("Moving")
                            current = current.children[prefix]
                            word = word[index:]
                            break

                        else:
                            logging.debug(f'Creating new node "{key_suffix}"')
                            is_word = len(prefix) == len(word)
                            logging.debug(
                                f'Creating new node "{prefix}", is it a word: {is_word}')
                            current.children[prefix] = AttributeNode(None, Children())
                            self.num_words += self.add_attributes(
                                current.children[prefix], attributes if is_word else None)
                            
                            # we know this is set to empty dict from above
                            current.children[prefix].children[key_suffix] = current.children[key] # type: ignore
                            logging.debug(f'Deleting old node "{key}"')
                            del current.children[key]

                        if len(word_suffix) > 0:
                            logging.debug("Iterate to add word remainder")
                            current = current.children[prefix]
                            word = word[index:]

                        break

                if not match_found:
                    logging.debug(
                        f'No overlapping prefixes in any key, adding "{word}" with `attributes` = {attributes}')
                    if current.children != None:
                        current.children[word] = AttributeNode()
                        self.num_words += self.add_attributes(
                            current.children[word], attributes)
                    break

    def delete(self, word: str):
        NotImplemented()

    def add_words(self, words: Words):
        """
            This function is to speed up initialzing a Trie by using a sorted collection of words.
            It also allows for storing subtries to a file based on a desired depth while building, allowing you
            to build very large tries without consuming too much memory. This is a trade-off for speed for lookups and number of files
            (greater depth, more files, but faster loading of subtries on disk), however can create more files, so only use
            if necessary.

            NOTE: the words passed in MUST be in lexigraphically sorted order, or else the output will not be correct
        """
        self._add_words_recursive(words, self.root, 0, 0)

    def get_matching_prefixes(self, words: Words, offset: int) -> tuple[list[Word], Optional[Word]]:
        """
            Returns a list of words that have at least their first character in 
            common, and the first non-matching one so we can add it back into the generator
        """
        logging.debug(">> get_matching_prefixes")
        logging.debug(f"Offset: {offset}")
        if logging.getLogger().level == logging.DEBUG:
            words_copy = tee(words)
            logging.debug(f"Words: {list(words_copy)}")
        matches: list[Word] = []
        first = next(words, None)
        logging.debug(f"First: {first}")
        if first == None:
            return (matches, None)

        matches.append(first)
        first_label = _get_values(first)[0]
        logging.debug(f"First label: {first_label}")
        first_label = first_label[offset:]

        # TODO: compare with takewhile
        current = next(words, None)
        while current != None:
            current_label = _get_values(current)[0]
            logging.debug(f"Current label: {current_label}")
            current_label = current_label[offset:]
            logging.debug(first_label)
            logging.debug(current_label)
            try:
                if first_label[0] == current_label[0]:
                    logging.debug("Matching prefix")
                    matches.append(current)
                    current = next(words, None)
                else:
                    logging.debug("No matching prefix")
                    break
            except IndexError:
                # TODO: fix this, it seems to be when a following word is shorter than the first?
                logging.warn(first_label, current)
                current = next(words, None)

        logging.debug(f"Matches: {matches}, last: {current}")
        logging.debug("<< get_matching_prefixes")
        # return it as a list so we can use len() on it
        return (matches, current)

    def _add_words_recursive(self, words: Words, current: AttributeNode, offset: int, depth: int):
        """
          Pure recursive method for adding sorted list of words.

          `words` must be an iterator so that `next` is available

          TODO: potentially could create subtries in parallel to speed it up
        """

        while 1:
            # `matches` will contain all words with at least one matching prefix
            # `last` contains the first 'peeked' word which didn't match
            matches, last = self.get_matching_prefixes(words, offset)

            # base case
            if len(matches) == 0:
                logging.debug("No matches - base case")
                return

            words = cast(Words, chain([last], words))

            if (current.children == None):
                current.children = {}

            first_label, first_attributes = _get_values(matches[0])
            first_label_copy = first_label
            first_label = first_label[offset:]

            # matching case, add to Trie
            if len(matches) == 1:
                logging.debug("Single match - normal case")
                logging.debug(
                    f"Adding word {first_label} with attributes {first_attributes}")
                current.children[first_label] = AttributeNode()
                self.num_words += self.add_attributes(
                    current.children[first_label], first_attributes)
                self.post_add_node(node = current, label = first_label_copy, prefix = '', depth = depth, max_length = 0)
                if last == None:
                    return
                continue

            # recursive case
            else:
                logging.debug("Multiple matches")

                # test if they are all the same word
                # since it's a sorted list, we can do this by comparing the first
                # and last words
                last_label = _get_values(matches[-1])[0]
                last_label = last_label[offset:]
                logging.debug(f"{first_label} vs {last_label}")
                prefix_length = get_longest_prefix_index(
                    first_label, last_label)
                prefix = first_label[:prefix_length]
                logging.debug(f"Prefix: {prefix} ({prefix_length})")

                # create a node for the longest matching prefix...
                # the next step decides if it's a word or not
                current.children[prefix] = AttributeNode()

                matches = iter(matches)

                # if the length of the longest prefix is the same as the first word, it's a word
                if (prefix_length == len(first_label)):
                    logging.debug("Prefix is the first word")

                    # there could be multiple instances of the word, so keep adding all matching ones
                    word = next(matches, None)
                    while word != None:
                        label, attributes = _get_values(word)
                        if label[offset:] != prefix:
                            break
                        logging.debug(
                            f"Adding word {label} with attributes {attributes}")
                        self.num_words += self.add_attributes(
                            current.children[prefix], attributes)
                        word = next(matches, None)

                    # if it's the end of the iter we're done
                    if word != None:
                        matches = chain([word], matches)

                # otherwise, it's just a node
                else:
                    logging.debug("Prefix is not a word")

                logging.debug("Recursing...")
                self._add_words_recursive(
                    matches, current.children[prefix], offset + prefix_length, depth + 1)
                self.post_add_node(node = current, label = first_label_copy[:offset + prefix_length], prefix = prefix, depth = depth, max_length = 0)
  
            if last == None:
                break


    def words(self, sort: bool =False) -> Iterator[str]:
        """
        Returns all nodes that are words
        """
        for prefix, node in self.nodes(sort):
            if (node.attributes != None):
                yield prefix

    def nodes(self, sort: bool = False) -> Iterator[Record]:
        prefix = ""
        stack: deque[Entry] = deque()
        if self.root.children != None:
            items = self.root.children.items()
            if sort:
                items = reversed(sorted(items))
            for item in items:
                stack.append(item)

        while stack:
            prefix, node = stack.pop()

            yield (prefix, node)

            if node.children != None:
                items = node.children.items()
                if sort:
                    items = reversed(sorted(items))
                for key, value in items:
                    stack.append((prefix + key, value))

    def _get_node(self, word: str) -> tuple[Record, AttributeNode | None] | None:
        return self._get_node_recursive(self.root, None, word, "")

    def _get_node_recursive(self, node: AttributeNode, previous_node: Optional[AttributeNode], word: str, prefix: str) -> Optional[tuple[Entry, Optional[AttributeNode]]]:
        if node.children != None:
            if (word in node.children):
                return ((prefix + word, node.children[word]), previous_node)

            # try seeing if there's a node with a matching prefix starting from shortest to longest
            for i in range(0, len(word)):
                w = word[:i]
                if w in node.children:
                    logging.debug(f"Prefix: {prefix}, Word: {word}")
                    return self._get_node_recursive(node.children[w], node, word[i:], prefix + w)

            return None
        return None

    # TODO: this returns a generator which technically isn't correct, but works for most cases
    def items(self) -> ItemsView[str, Attributes]:
        """
            Method to make it interoperable with dict, where keys are the words, and values are the attributes
        """
        
        for prefix, node in self.nodes():
            if (node.attributes != None):
                yield (prefix, node.attributes)

    def search(self, word, type: Literal['fuzzy', 'edit'] = 'edit', threshold: int | float = 0) -> Candidates:

        # Pruning phase
        candidates = []
        offset = 0
        distance = 0

        if type == 'edit':
          self._search_edit_recursive(self.root, word, "", offset, distance, int(threshold), candidates)
        else:
          self._search_fuzzy_recursive(self.root, word, "", offset, distance, threshold, candidates)

        return candidates

    def _search_edit_recursive(self, node: AttributeNode, word: str, prefix: str, offset: int, current_distance: int, max_distance: int, candidates: Candidates) -> Candidates:
        if node == None or node.children == None:
            return

        for child in node.children:
            # TODO: need to account for if fragment is smaller than child or vice versa
            new_offset = offset + len(child)
            fragment = word[offset:max(len(child), len(word) - offset)]
            d = distance(fragment, child)
            total = d + current_distance
            logging.debug(f"Distance b/w {child} and {fragment}: {d}")
            logging.debug(f"{current_distance} + {d} = {total}")
            
            if total <= max_distance:
                child_node = node.children[child]
                new_prefix = f"{prefix}{child}"
                if child_node.attributes != None:
                    # compute distance to the remainder of the word
                    total_distance = len(word) - len(new_prefix) + d
                    logging.debug(f"{word}, {new_prefix}, {total_distance}")
                    if total_distance <= max_distance:
                        candidates.append((total_distance, new_prefix, child_node))
                self._search_edit_recursive(child_node, word, new_prefix, new_offset, total, max_distance, candidates)

    def _search_fuzzy_recursive(self, node: AttributeNode, word: str, prefix: str, offset: int, current_distance: int, threshold: float, candidates: Candidates) -> Candidates:
        if node == None or node.children == None or len(node.children) == 0:
            return
        
        # TODO: investigate if doing a running similarity is faster than skipping early ones
        
        for child in node.children:
            new_prefix = f"{prefix}{child}"

            min_sim = abs((len(new_prefix) - len(word)) / len(word))

            # if theres' too many letters difference, there will be no more matches
            if min_sim > threshold:
                return
            
            # if there's not enough letters, we still need to recurse to check them
            elif min_sim < 0:
                self._search_fuzzy_recursive(child_node, word, new_prefix, offset + len(child), None, threshold, candidates)  

            # within the length range where you could have similarities, need to compare.
            # NOTE: even if a word is too short, 
            else:
              r = ratio(word, new_prefix)
              logging.debug(f"Search word: {word}, Prefix: {new_prefix}, Ratio: {r}")
              if r >= threshold:
                  child_node = node.children[child]
                  if child_node.attributes != None:
                      candidates.append((r, new_prefix, child_node))
                  self._search_fuzzy_recursive(child_node, word, new_prefix, offset + len(child), None, threshold, candidates)  

    def stats(self, unique: bool = True):
        average_length: int = 0
        word_lengths: dict[int, int] = {}
        letter_frequency: dict[str, int] = {}
        letter_distribution: dict[str, dict[int, int]] = {}
        num_nodes: int = 0
        node_distribution: dict[int, int] = {}
        lengths_at_node_depths: dict[int, dict[int, int]] = {}
        depth: int = 0

        # initialize the stack
        prefix = ""
        stack: deque[tuple[str, AttributeNode, int]] = deque()
        if self.root.children != None:
            items = self.root.children.items()
            node_distribution[1] = len(items)
            for key, value in self.root.children.items():
                stack.append((key, value, 1))

        while stack:
            prefix, node, depth = stack.pop()

            num_nodes += 1

            # do the stats...
            lengths_at_node_depths[depth] = lengths_at_node_depths.get(
                depth, {})
            lengths_at_node_depths[depth][len(
                prefix)] = lengths_at_node_depths[depth].get(len(prefix), 0) + 1

            if node.attributes != None:
                # it's a word, so 'prefix' is the full word
                length = len(prefix)

                count = 1 if unique else self.count_attributes(node)

                # TODO: just compute afterwards based on word lengths?
                average_length += length * count
                word_lengths[length] = word_lengths.get(length, 0) + count

                for index, letter in enumerate(prefix):
                    letter_frequency[letter] = letter_frequency.get(
                        letter, 0) + count
                    letter_distribution[letter] = letter_distribution.get(
                        letter, {})
                    letter_distribution[letter][index] = letter_distribution[letter].get(
                        index, 0) + count

            if node.children != None:
                items = node.children.items()
                node_distribution[depth + 1] = node_distribution.get(depth + 1, 0) + len(items)
                for key, value in items:
                    stack.append((prefix + key, value, depth + 1))

        return {
            'num_words': self.num_words,
            'average_length': average_length / self.num_words,
            'word_lengths': word_lengths,
            'letter_frequency': letter_frequency,
            'letter_distribution': letter_distribution,
            'num_nodes': num_nodes,
            'node_distribution': node_distribution,
            'lengths_at_node_depths': lengths_at_node_depths
        }

def levenstein(self, word1, word2):
    length = len(word2) + 1
    previous_row = array('b', range(length))
    for i in range(len(word1)):
        current_row = array('b', range(length))
        current_row[0] = i + 1
        for j in range(length - 1):
            deletion_cost = previous_row[j + 1] + 1
            insertion_cost = current_row[j] + 1
            substitution_cost = previous_row[j] if word1[i] == word2[j] else previous_row[j] + 1
            current_row[j + 1] = min(deletion_cost,
                                     insertion_cost, substitution_cost)

        previous_row = current_row
    return previous_row[length - 1]
