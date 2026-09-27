import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("..")

from data.dataloaders import read_df_file
from env_setup import LocalVariables

lv = LocalVariables("../local_config.yaml")

data_df, _ = read_df_file(lv.source_class)

list_all = data_df['finding_labels'].tolist()
list_all = list(map(str.lower, list_all))
class_type_counts = {}
for i, x in enumerate(list_all):
    for y in x.split('|'):
        if y not in class_type_counts:
            class_type_counts[y] = 1
        else:
            class_type_counts[y] += 1

class_names = list(class_type_counts.keys())
class_counts = list(class_type_counts.values())

fig = plt.figure(figsize = (12,12))

plt.bar(class_names, class_counts, color='blue', width=0.4)
for i, (n, c) in enumerate(zip(class_names, class_counts)):
    plt.text(i, c, c, ha='center')

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.xlabel("Lung Disease")
plt.ylabel("Number of patients with Lung Disease")
plt.title("Patients with Lung Diseases")
plt.savefig("class_type_distribution.png")