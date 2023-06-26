import logging
import os
import pickle
import gc
from itertools import chain, tee
from collections import deque
from collections.abc import MutableMapping
from array import array
from typing import Any, Callable, ItemsView, Iterator, Optional, TypeAlias, cast
from rapidfuzz import fuzz
# from rapidfuzz.distance.Levenshtein import distance

Word: TypeAlias = str | tuple[str, Any]
Words: TypeAlias = Iterator[Word]
Attributes: TypeAlias = Any
Entry: TypeAlias = tuple[str, 'Node']
Record: TypeAlias = tuple[str, Attributes]
Children: TypeAlias = dict[str, 'Node']

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
        Returns the length of the longest prefix in a sorted list of words
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
    try:
        return (element[0], element[1]) if isinstance(element, tuple) else (element, True)
    except IndexError:
        print(element[0])
        return ("null", -1)

class Node:
    # avoid storing class attributes in '__dict__' to save memory
    __slots__ = ('attributes', 'children')

    def __init__(self, attributes: Attributes = None, children: Optional[Children] = None):
        self.attributes = attributes
        self.children = children

    def __str__(self):
        return self.attributes

    def print(self, depth: int):
        if self.children == None:
            return ""
        offset = "\t"*depth
        string = ""
        for item in self.children.items():
            key: str = item[0]
            child: Node = item[1]
            string += "\n" + offset + "\t" + \
                f"{key}({child.attributes}): {child.print(depth+1)}"
        return string

class StoredNode(Node):
    __slots__ = ('attributes', 'children', 'saved')

class Trie(MutableMapping[str, Attributes]):
    def __init__(self, 
                 root: Node | None=None, 
                 add_word: Optional[Callable[[Node, Attributes], int]]=None, 
                 delete_word: Optional[Callable[[Node], int]]=None, 
                 count: Optional[Callable[[Node], int]]=None, 
                 depth_to_store: Optional[int]=None, 
                 words: Optional[Words] = None, 
                 subtrie_path: str='./subtrie'):
        """
            Initialize a Trie.

            NOTE: `words` must be either a list of strings or a list of tuples of the form (word: str, attributes: Any). If `words` is a list of
            strings, the `attributes` property will be set to `True` for each word.

            You can pass in a custom add function that is used to control how attributes are defined. This can be useful if you need
            special cases for when words conflict or requiring merging of attributes, i.e. if `attributes` is an object and you add
            a word that already exists with other attributes, you may want to overwrite, merge, etc. The default is to simply assign
            the value `True`, indicating it is a word, and adding the same word again will not affect the trie.

            NOTE: if you supply custom add_word and delete_word functions, they MUST manage the `num_words` property if you want `len(trie)` to report the correct value.
        """

        self.root = root if root != None else Node()
        self._add = add_word if add_word != None else self._default_add
        self._delete = delete_word if delete_word != None else self._default_delete
        self._count = count if count != None else self._default_count
        self.num_words: int = 0
        self.depth_to_store = depth_to_store
        self.subtrie_path = subtrie_path
        if self.depth_to_store != None:
            os.mkdir(self.subtrie_path)
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

    def _default_add(self, node: Node, value: Attributes) -> int:
        """
          The default add method to use when one isn't provided. It uses True/None to indicate whether a node is a word or not
        """
        # if this is a new word, increment the number of words
        # otherwise we are just overwriting attributes which isn't a new word
        is_new = True
        if node.attributes != None:
            is_new = False
        node.attributes = value
        return 1 if is_new else 0

    def _default_delete(self, node: Node) -> int:
        node.attributes = None
        return -1

    def _default_count(self, node: Node) -> int:
        return 1

    def add(self, word: str, attributes: Attributes=True):
        """
          Adds a single word to an already constructed Trie. You should use `add_words` to initialize a Trie for performance reasons
        """
        current: Node = self.root

        while True:
            logging.debug(f'Adding "{word}"')

            if current.children == None:
                logging.debug("No children, initializing")
                current.children = cast(Children, {})

            logging.debug(str(current.children.keys()))

            if len(current.children.keys()) == 0:
                logging.debug(
                    f'Empty children, adding "{word}" with `attributes = {attributes}')
                current.children[word] = Node()
                self.num_words += self._add(current.children[word], attributes)
                break

            elif word in current.children.keys():
                logging.debug(
                    f'"{word}" already exists, adding attributes {attributes}')
                self.num_words += self._add(current.children[word], attributes)
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
                            current.children[prefix] = Node(None, Children())
                            self.num_words += self._add(
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
                        current.children[word] = Node()
                        self.num_words += self._add(
                            current.children[word], attributes)
                    break

    def delete(self, word: str):
        prev: Node | None = None
        current = self.root

        while current != None:
            if word in current.children:
                # handle node shifts

                self._delete(node)
                return

            for i in range(0, len(word)):
                w = word[:i]
                if w in node.children:
                    logging.debug(f"Prefix: {prefix}, Word: {word}")
                    prev = current
                    current = current.children[w]

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
                # TODO: fix this, it seems to be when a following word is shorted than the first?
                logging.warn(current)
                current = next(words, None)

        logging.debug(f"Matches: {matches}, last: {current}")
        logging.debug("<< get_matching_prefixes")
        # return it as a list so we can use len() on it
        return (matches, current)

    def _add_words_recursive(self, words: Words, current: Node, offset: int, depth: int):
        """
          Pure recursive method for adding list of sorted words.

          `words` must be an iterator so that `next` is available

          TODO: could create subtries in parallel to speed it up
        """

        while 1:
            # `matches` will contain all words with at least one matching prefix
            # `last` contains the first 'peeked' word which didn't match
            matches, last = self.get_matching_prefixes(words, offset)

            # base case
            if len(matches) == 0:
                logging.debug("No matches - base case")
                return

            # handle the matches, then continue where it left off
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
                current.children[first_label] = Node()
                self.num_words += self._add(
                    current.children[first_label], first_attributes)
                if last == None:
                    return
                continue

            # recursive case
            else:
                logging.debug("Multiple matches")

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
                current.children[prefix] = Node()

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
                        self.num_words += self._add(
                            current.children[prefix], attributes)
                        word = next(matches, None)

                    if word != None:
                        matches = chain([word], matches)

                # otherwise, it's just a node
                else:
                    logging.debug("Prefix is not a word, just recurse")

                logging.debug("Recursing...")
                self._add_words_recursive(
                    matches, current.children[prefix], offset + prefix_length, depth + 1)

                # dump to storage if specified
                if (self.depth_to_store != None and depth == self.depth_to_store):
                    filename = get_filename(first_label_copy)
                    logging.info(
                        f"Moving trie at '{first_label}' as '{filename}' to file")
                    # TODO: investigate mmap instead of pickling
                    with open(f"{self.subtrie_path}/{filename}.pickle", 'wb') as file:
                        pickle.dump(current, file)
                    # cleanup memory
                    del current.children[prefix]
                    gc.collect()

            if last == None:
                break

        logging.debug("End")

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

    def _get_node(self, word: str) -> tuple[Record, Node | None] | None:
        return self._get_node_recursive(self.root, None, word, "")

    def _get_node_recursive(self, node: Node, previous_node: Optional[Node], word: str, prefix: str) -> Optional[tuple[Entry, Optional[Node]]]:
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

    # TODO: this returns a generator which technically isn't correct
    def items(self) -> ItemsView[str, Attributes]:
        """
            Method to make it interoperable with dict, where keys are the words, and values are the attributes
        """
        
        for prefix, node in self.nodes():
            if (node.attributes != None):
                yield (prefix, node.attributes) # type: ignore

    def search1(self, word, threshold):
        candidates = []

        stack = deque()
        if self.root.children != None:
            for key, node in self.root.children.items():
                stack.append((key, node))

        while stack:
            prefix, node = stack.pop()
            similarity = fuzz.ratio(word[:len(prefix)], prefix)
            # logging.debug(f"{word[:len(prefix)]} <> {prefix}: {similarity}")
            if similarity >= threshold:
                if node.attributes != None:
                    candidates.append((similarity, prefix, node))

                if node.children != None:
                    for key, value in reversed(sorted(node.children.items())):
                        stack.append((prefix + key, value))

        return candidates

    def search(self, word, max_distance: int):

        # Pruning phase
        candidates = []

        stack = deque()
        if self.root.children != None:
            for key, node in self.root.children.items():
                stack.append((key, node))

        while stack:
            prefix, node = stack.pop()
            prefix_length = len(key)

            prefix = word[:prefix_length]
            logging.debug(f"Comparing {prefix} to {word}")
            previous_row = array('b', range(prefix_length + 1))
            for i in range(len(prefix) - 1):
                current_row = array('b', range(prefix_length + 1))
                current_row[0] = i + 1
                for j in range(prefix_length - 1):
                    deletion_cost = previous_row[j + 1] + 1
                    insertion_cost = current_row[j] + 1
                    substitution_cost = previous_row[j] if word[i] == prefix[j] else previous_row[j] + 1
                    current_row[j + 1] = min(deletion_cost,
                                             insertion_cost, substitution_cost)

                previous_row = current_row

            distance = previous_row[prefix_length - 1]
            logging.debug(f"Distance: {distance}")
            if previous_row[prefix_length - 1] < max_distance:
                if node.children != None:
                    for key, value in node.children.items():
                        stack.append((prefix + key, value))

        return candidates

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
        stack: deque[tuple[str, Node, int]] = deque()
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

                count = 1 if unique else self._count(node)

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
                node_distribution[depth +
                                  1] = node_distribution.get(depth + 1, 0) + len(items)
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