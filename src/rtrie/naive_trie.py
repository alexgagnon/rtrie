import sys

class NaiveTrie:
    __slots__ = ('root',)

    def __init__(self, words = None):
        self.root = Node()
        if words != None:
            for word in words:
                self.add(word)
    
    def items(self):
        stack = [(self.root, '')]

        while stack:
            current, prefix = stack.pop()
            if current.attributes != None:
                yield (prefix + key, current.attributes)
            if current.children != None:
                for key, child in current.children.items():
                    stack.append((child, prefix + key))

        
    def __len__(self):
        return len(list(self.items()))
    
    def __contains__(self, word):
        return self.search(word)
    
    def _get_node(self, word: str):
        current = self.root
        for letter in word:
            if current.children == None:
                return None
            if letter not in current.children:
                return None
            current = current.children[letter]
        return (current, word)
    
    def __getitem__(self, word: str):
        result = self._get_node(word)
        if result != None and result[0][1].attributes != None:
            return (result[0][0], result[0][1].attributes)
        return None

    def __setitem__(self, word: str, attributes) -> None:
        result = self._get_node(word)
        if result != None:
            result[0][1].attributes = attributes

    def __repr__(self):
        return self.root.print(0)

    def __contains__(self, word: object) -> bool:
        for letter in word:
            if letter not in self.root.children:
                return False
            self.root = self.root.children[letter]
        return True
    
    def remove(self, word):
        current = self.root
        prev = None
        prev_key = None
        for letter in word:
            if current.children == None:
                return
            if letter not in current.children:
                return
            prev = current
            prev_key = letter
            current = current.children[letter]
        if current.children != None:
            del prev.children[prev_key]
        else:
            current.attributes = None
        
    def add(self, word, attributes = True):
        current = self.root
        for letter in word:
            letter = sys.intern(letter)
            if current.children == None:
                current.children = {}
            if letter not in current.children:
                current.children[letter] = Node()
            current = current.children[letter]
        current.attributes = attributes
    
class Node():
    __slots__ = ('attributes', 'children')

    def __init__(self, attributes = None, children = None, *args, **kwargs):
        self.attributes = attributes
        self.children = children

    def print(self, depth: int):
        if self.children == None:
            return ""
        offset = "\t" * depth
        string = ""
        for key, child in self.children.items():
            string += "\n" + offset + "\t" + \
                f"{key}({child.attributes}): {child.print(depth+1)}"
        return string