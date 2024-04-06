from typing import cast
import sys

class NaiveTrie:
    def __init__(self, words):
        self.root = Node()
        self.num_words: int = 0
        for word in words:
            self.add(word)
    
    def __contains__(self, word):
        return self.search(word)
    
    def __getitem__(self, word: str):
        result = self._get_node(word)
        if result != None and result[0][1].attributes != None:
            return (result[0][0], result[0][1].attributes)
        return None

    def __setitem__(self, word: str, attributes) -> None:
        result = self._get_node(word)
        if result != None:
            result[0][1].attributes = attributes

    def __str__(self):
        return self.root.print(0)

    def __contains__(self, word: object) -> bool:
        for letter in word:
            if letter not in self.root.children:
                return False
            self.root = self.root.children[letter]
        return True

    def __iter__(self):
        return self.words()
    
    def post_add_node(self, **kwargs):
        """
        Used to hook into the add method to perform additional operations after a node is added.
        By default it's a no-op.
        """
        pass
    
    def add(self, word):
        current = self.root
        for i, letter in enumerate(word):
            letter = sys.intern(letter)
            if current.children == None:
                current.children = {}
            if letter not in current.children:
                current.children[letter] = Node()
            current = current.children[letter]
        current.attributes = True
    
class Node():
    __slots__ = ('attributes', 'children')

    def __init__(self, attributes = None, children = None, *args, **kwargs):
        self.attributes = attributes
        self.children = children

    def __str__(self):
        return self

    def print(self, depth: int):
        if self.children == None:
            return ""
        offset = "\t" * depth
        string = ""
        for key, child in self.children.items():
            string += "\n" + offset + "\t" + \
                f"{key}({child.attributes}): {child.print(depth+1)}"
        return string