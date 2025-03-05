from helpers import *

# Load data
data_dict = load_data_pairs()

# Print the keys in data_dict
print("Keys in data_dict:")
print(data_dict.keys())

# Print number of entries for each key
print("\nNumber of entries per key:")
for key in data_dict:
    print(f"{key}: {len(data_dict[key])}")

# Get samples from pos and neg pairs
pos_pairs = flatten_list([data_dict[x] for x in data_dict if "pos" in x])
neg_pairs = flatten_list([data_dict[x] for x in data_dict if "neg" in x])

print(f"\nTotal positive pairs: {len(pos_pairs)}")
print(f"Total negative pairs: {len(neg_pairs)}")

# Print some example pairs
print("\nExample positive pairs:")
for pair in pos_pairs[:3]:
    print(pair)

print("\nExample negative pairs:")
for pair in neg_pairs[:3]:
    print(pair)