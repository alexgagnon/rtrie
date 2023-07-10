from typing import Any
from .trie import Trie, Node

class StringTrie(Trie):
    def __init__(self, separator: str = "|", **kwargs):
        self.separator = separator
        super().__init__(**kwargs)

    def add_attributes(self, node: Node, value: Any) -> int:
        if value == None:
            node.attributes = None
            return 0
        if node.attributes == None:
            node.attributes = str(value)
            return 1
        else:
            # check if it already exists or not
            values = node.attributes.split(self.separator)
            if value in values:
                return 0
            node.attributes = f"{node.attributes}{self.separator}{str(value)}"
            return 1

    def delete_attributes(self, node, value):
        # TODO
        NotImplemented()        

    def count_attributes(self, value):
        return len(value.split(self.separator)) if value != None else 0