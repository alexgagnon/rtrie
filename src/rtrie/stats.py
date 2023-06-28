import json

def compute_stats(file):
    with open(file) as f:
        stats = json.load(f)
        print(stats['average_length'])
        print(stats['num_words'])
        print(stats['num_nodes'])
        print(get_median(stats['word_lengths'], stats['num_words']))

def get_mode(frequencies):
    if frequencies == None or len(frequencies) == 0:
       return None
    most = 0
    modes = []
    for key, value in frequencies.items():
      if value > most:
          modes = [key]
          most = value
      elif value == most:
          modes.append(key)
    return modes

def get_median(frequencies, total = None):
    if frequencies == None or len(frequencies) == 0:
       return None
    frequencies = {key:frequencies[key] for key in sorted(frequencies.keys(), key=int)}
    print(frequencies)
    if total == None:
      total = sum(frequencies.values())
    
    half = total / 2

    current = 0
    for key, value in frequencies.items():
        current += value
        if current >= half:
          return key