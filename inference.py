import matplotlib.pyplot as plt
import numpy as np

data = [0.45, 35.59, 11.23]
data = [(v / 0.45) * 100 for v in data]

print(data)

labels = ["PyTorch CPU", "PyTorch cuda", "burn wgpu"]
ind = np.arange(len(data))

fig, ax = plt.subplots()

ax.grid(axis="y")
ax.set_axisbelow(True)

ax.bar(ind, data, 0.35)
ax.set_xticks(ind)
ax.set_xticklabels(labels=labels)

ax.set_ylabel("Токенов в секунду")

plt.savefig("inference_ru.png")