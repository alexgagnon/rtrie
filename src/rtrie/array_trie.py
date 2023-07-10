import collections
from typing import Any
from .trie import Trie, Node

class ArrayTrie(Trie):
    def __init__(self, no_array_for_single_value: bool = False, **kwargs):
        self.no_array_for_single_value = no_array_for_single_value
        super().__init__(**kwargs)
        
    def add_attributes(self, node: Node, value: Any) -> int:
        if value == None:
            node.attributes = None
            return 0
        if node.attributes == None:
            node.attributes = value if self.no_array_for_single_value else [value]
            return 1
        else:
            # check if it already exists or not
            if isinstance(node.attributes, collections.abc.Sequence):
                if value in node.attributes:
                    return 0
                
                node.attributes.append(value)
                return 1
            else:
                node.attributes = [node.attributes, value]
                return 1

    def delete_attributes(self, node, value):
        # TODO
        NotImplemented()

    def count_attributes(self, value):
        return len(value) if value != None else 0