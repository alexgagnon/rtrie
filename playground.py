from rtrie import Trie
import logging
import os

level = logging.DEBUG if os.environ.get("DEBUG") == "True" else logging.INFO
logging.basicConfig(level = level)

def add_attributes(collection, value):
    if value == None:
        logging.debug("No attributes to add")
        collection.attributes = None
        return 0
    else:
      if collection.attributes == None:
          logging.debug("Creating attributes")
          collection.attributes = str(value)
      else:
          # using .join and interpolation is faster than = or +=
          logging.debug("Appending attributes")
          collection.attributes = f"{collection.attributes}|{str(value)}"
      return 1

words = ['Hello', 'Hey', 'Man', 'Man', 'Mani', 'Manilla', 'Manitee', 'There']
words = [tuple([word, i]) for i, word in enumerate(words)]
words = (w for w in words)
trie = Trie(words=words, add_word=add_attributes)

print(trie)
print('Mani' in trie)
candidates = trie.search('Manilla', max_distance=3)
print([(x[0], x[1].attributes) for x in candidates])