###
# This module contains the Node classes that are used to build the Trie.
# They use slots to reduce memory usage and improve performance.
# However, slots can't be used in a diamon inheritance, so need to carefully
# plan the inheritance hierarchy. For example, with the MaxLengthStringAttributeNode,
# we intentionally left the MaxLengthNode's slots empty to avoid layout conflicts,
# and then added the necessary slot in the MaxLengthStringAttributeNode.
###

from .types import Attributes
from typing import Any, Optional, TypeAlias

Entry: TypeAlias = tuple[str, 'AttributeNode']
Children: TypeAlias = dict[str, 'AttributeNode']
Candidates: TypeAlias = list[tuple[int, str, 'AttributeNode']]

class Node:
    """
      Base Abstract Node type
    """
    __slots__ = ()

class AttributeNode(Node):
    """
      Node that supports attributes
    """
    __slots__ = ('attributes', 'children')

    attributes: Attributes
    children: Optional[Children]

    def __init__(self, attributes: Attributes = None, children: Optional[Children] = None, *args, **kwargs):
        self.attributes = attributes
        self.children = children
        super().__init__(*args, **kwargs)
    
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

    def delete_attributes(self) -> int:
        deleted = 0
        if self.attributes != None:
          self.attributes = None
          deleted = -1
        return deleted

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

    def count_attributes(self, value):
        return len(value.split(self.separator)) if value != None else 0

class MaxLengthStringAttributeNode(MaxLengthNode, StringAttributeNode):
    __slots__ = ('max_length')

    def __init__(self, attributes: Attributes = None, children: Optional[Children] = None, max_length: int = 0, *args, **kwargs):
        super().__init__(attributes=attributes, children=children, max_length=max_length, *args, **kwargs)
