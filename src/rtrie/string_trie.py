from .trie import Trie
from .node import StringAttributeNode

class StringTrie(Trie):
    def __init__(self, separator: str = "|", **kwargs):
        StringAttributeNode.separator = separator
        self.node_factory = StringAttributeNode
        super().__init__(**kwargs)
