import pandas as pd
import sys
from collections import Counter

filepath = sys.argv[1]
df = pd.read_csv(filepath, sep='\t', header=0)

