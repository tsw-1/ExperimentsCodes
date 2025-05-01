import os
def natural_sort_key(s):
    """Sort key for natural sort."""
    parts = []
    number = ''
    for char in s:
        if char.isdigit():
            number += char
        else:
            if number:
                parts.append(int(number))
                number = ''
            parts.append(char.lower())
    if number:
        parts.append(int(number))
    return parts

def natsorted(iterable):
    """Sort an iterable in natural order."""
    return sorted(iterable, key=natural_sort_key)

# Example usage
# example_list = os.listdir('/E22301339/HAN-BERT/eRisk2017/processed/combined_maxsim16/train')
# sorted_list = natsorted(example_list)
# print(sorted_list)  # Output: ['file1.txt', 'file2.txt', 'file3.txt', 'file10.txt', 'file11.txt']
