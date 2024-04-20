###
# This module contains the Node classes that are used to build the Trie.
# They use slots to reduce memory usage and improve performance.
# However, slots can't be used in a diamon inheritance, so need to carefully
# plan the inheritance hierarchy. For example, with the MaxLengthStringAttributeNode,
# we intentionally left the MaxLengthNode's slots empty to avoid layout conflicts,
# and then added the necessary slot in the MaxLengthStringAttributeNode.
# NOTE: need to make sure that ALL classes that inherit from Node have the __slots__
# attribute defined, even if it's empty, otherwise they will be __dict__ based and
# consume more memory.
###

from collections import deque
from .types import Attributes
from typing import Any, Optional, TypeAlias, TypedDict
from abc import ABC
import sys

Entry: TypeAlias = tuple[str, 'Node']
Children: TypeAlias = dict[str, 'Node']
Candidates: TypeAlias = list[tuple[int, str, 'Node']]

class Node(ABC):
    """
      Base Node type
    """
    __slots__ = ('children',)
    
    def __init__(self, children):
        self.children = children

class AttributeNode(Node):
    """
      Node that supports attributes
    """
    __slots__ = ('attributes',)

    def __init__(self, attributes: Attributes = None, children: Optional[Children] = None, *args, **kwargs):
        self.attributes = attributes
        super().__init__(children, *args, **kwargs)
    
    def __repr__(self):
        return f"(Attributes: {self.attributes}, Children: {self.children.keys() if self.children != None else None})"
    
    def is_word(self) -> bool:
        return self.attributes != None
    
    def nodes(self, include_root = False, sort: bool = False, prefix: str = ""):
        """
        BFS through child nodes. If sort is True, then sort the children in reverse order. If a prefix is provided,
        prepend it to each yielded key.

        TODO: investigate if it's faster to just have the consumer sort after the fact instead of sorting here.
        """
        stack: deque[Entry] = deque([("", self)] if include_root else [])
        if self.children != None:
            items = self.children.items()
            if sort:
                items = reversed(sorted(items))
            for item in items:
                stack.append(item)

        while stack:
            p, node = stack.pop()

            yield (prefix + p, node)

            if node.children != None:
                items = node.children.items()
                if sort:
                    items = reversed(sorted(items))
                for key, value in items:
                    stack.append((p + key, value))
   
    def items(self, include_root = False, prefix: str = ""):
        """
        Returns a generator that yields each word. Providing a prefix will prepend it to each word,
        and include_root will include the root node in the results if it's a word
        """
        for p, node in self.nodes(include_root = include_root, prefix = prefix):
            if (node.attributes != None):
                yield (p, node.attributes)

    def add_attributes(self, value: Attributes) -> int:
        """
          The default add method to use when one isn't provided. It adds to 'attributes' if defined, 
          to indicate whether a node is a word or not. The return value is used to keep a running 
          tally of how many words are in the Trie.

          The default for Trie is to set it to True, and num_words is only incremented if it's a new word.
        """
        # if this is a new word, increment the number of words
        # otherwise we are just overwriting attributes which isn't a new word
        is_new = 1
        if self.attributes != None:
            is_new = 0
        self.attributes = value
        return is_new
    
    def delete_attributes(self, value=None) -> bool:
        if self.attributes != None:
          self.attributes = None
          return  True
        return False

    def count_attributes(self) -> int:
        return 1 if self.attributes != None else 0

    def print(self, depth: int):
        if self.children == None:
            return ""
        offset = "\t" * depth
        string = ""
        for key, child in self.children.items():
            string += "\n" + offset + "\t" + \
                f"{key}({child.attributes}): {child.print(depth+1)}"
        return string
    
# TODO: BooleanAttributeNode...

class MaxLengthNode(Node):
    """
      Node that supports a max length
    """
    __slots__ = ()

    max_length: int

    def __init__(self, max_length: int = 0, *args, **kwargs):
        self.max_length = max_length
        super().__init__(*args, **kwargs)
    
class StringAttributeNode(AttributeNode):
    """
      Node that supports storing attributes as a string split by a separator
    """
    __slots__ = ()

    separator = '|'

    def add_attributes(self, value: Any) -> int:
        if value == None:
            self.attributes = None # TODO: confirm if this is the correct behavior
            return 0
        if self.attributes == None:
            self.attributes = value
            return 1
        else:
            # check if it already exists or not
            values = self.attributes.split(self.separator)
            if value in values:
                return 0
            self.attributes = f"{self.attributes}{self.separator}{str(value)}"
            return 1
    
    def delete_attributes(self, value: Any) -> bool:
        if self.attributes == None:
            return False
        values = self.attributes.split(self.separator)
        if value in values:
            values.remove(value)
            if len(values) == 0:
                self.attributes = None
            else:
                self.attributes = self.separator.join(values)
            return True
        return False
    
    def count_attributes(self, value):
        return len(value.split(self.separator)) if value != None else 0

class MaxLengthStringAttributeNode(MaxLengthNode, StringAttributeNode):
    __slots__ = ('max_length',)

    def __init__(self, attributes: Attributes = None, children: Optional[Children] = None, max_length: int = 0, *args, **kwargs):
        super().__init__(attributes=attributes, children=children, max_length=max_length, *args, **kwargs)