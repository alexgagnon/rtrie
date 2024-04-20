import argparse
from pathlib import Path
from rtrie import Trie, StringTrie, ArrayTrie
from os.path import splitext
from pickle import dump

def main():
  """
    TODO: allow loading in a file of sorted entries and building a trie from that
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("path")
  parser.add_argument("-d", "--depth", type = int)
  parser.add_argument("-o", "--output", type = str)
  parser.add_argument("-s", "--separator", type = str)
  parser.add_argument("-t", "--type", type = str, choices = ["Trie", "StringTrie", "ArrayTrie"], default = "Trie")
  args = parser.parse_args()
  input = Path(args.path)

  match args.type:
    case 'StringTrie':
      init = StringTrie
    case 'ArrayTrie':
      init = ArrayTrie
    case 'Trie':
      init = Trie
  
  kwargs = {}

  if args.depth != None:
    kwargs['depth'] = args.depth
  if args.separator != None:
    kwargs['separator'] = args.separator

  with open(input) as f:
    _, ext = splitext(input)
    print("Generating the Trie. Make sure the input is in lexicographically sorted order!")
    match ext:
      case '.tsv':
        words = (tuple(word.strip().split("\t")) for word in f)
      case _:
        raise "Unsupported filetype"
      # TODO: more types of inputs (i.e. csv, json, etc.)
    trie = init(words = words, **kwargs)
    if args.output != None:
      output = Path(args.output)
      with open(output, 'wb') as o:
        dump(trie, o)
        print(f"Successfully dumped to {output}")
    else:
      print(trie)

if __name__ == "__main__":
  main()
