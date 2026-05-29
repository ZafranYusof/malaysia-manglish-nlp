import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'tests'))
from manglish_nlp import normalize
from benchmark_expanded import NORM_DATA

for text, expected in NORM_DATA:
    result = normalize(text)
    if result != expected:
        print(f'FAIL: "{text}"')
        print(f'  Expected: "{expected}"')
        print(f'  Got:      "{result}"')
        print()
