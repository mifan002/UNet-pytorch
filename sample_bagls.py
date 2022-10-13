import numpy as np
import pandas as pd


# Size of BAGLS training dataset
N = 55750

# Sample 10%
np.random.seed(42)
random_idx = np.random.randint(0, N, N // 10)

# Save image and mask IDs to csv file
df = pd.DataFrame()
df['Images'] = [str(i) + '.png' for i in random_idx]
df['Masks'] = [str(i) + '_seg.png' for i in random_idx]
df.to_csv('bagls_ids.csv', index=False)
