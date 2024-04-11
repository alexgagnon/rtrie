from . import Trie, Node, StoredNode, Attributes, Children, Words
import pickle
import logging
import gc
import os
from typing import Any, Callable, Optional

def get_filename(string: str) -> str:
    return "".join([x if x.isalnum() else "_" for x in string])

class StoredTrie(Trie):
    """
      A trie that stores subtrees in files. A special sentinel node is used to indicate that a subtree is stored in a given file.
      NOTE: there is a large trade-off between memory and speed. If you store subtrees in files, you will have to load them from disk,
      and the depth you define how many files you will have (low depth = fewer, larger files, high depth = more, smaller files).
      Could potentially speed this up by using mmap?
    """
    depth_to_store: Optional[int]
    subtrie_path: str

    def __init__(self, 
                 root: Node | None=None, 
                 words: Optional[Words] = None,
                 subtrie_path: str = "subtries",
                 depth_to_store: Optional[int] = None
                ):
        super().__init__(root, words)
        self.depth_to_store = depth_to_store
        self.subtrie_path = subtrie_path
    
    def post_add_node(self, **kwargs):
        label = kwargs["label"]
        current = kwargs["current"]
        if (self.depth_to_store != None and kwargs["depth"] == self.depth_to_store):
            filename = get_filename(label)
            logging.info(
                f"Moving trie at '{label}' as '{filename}' to file")
            with open(f"{self.subtrie_path}{os.sep}{filename}.pickle", 'wb') as file:
                pickle.dump(current, file)
            current = StoredNode(None, filename)

    def load_node(self, filename: str) -> Node:
        with open(f"{self.subtrie_path}{os.sep}{filename}.pickle", 'rb') as file:
            node = pickle.load(file)
        return node
    
    def nodes(self):
        for node in super().nodes():
            if node.filename != None:
                node = self.load_node(node.filename)
            yield node
    