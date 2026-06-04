import matplotlib.pyplot as plt
import numpy as np

data = [
    {
        "min": 18.867887,
        "max": 24.10779,
        "mean": 20.795994,
        "median": 20.816633,
        "samples": 5
    },
    {
        "min": 5.300518200034276,
        "max": 6.465210799942724,
        "mean": 5.636665679980069,
        "median": 5.432073100004345,
        "samples": 5
    },
    {
        "min": 10.118615,
        "max": 14.2938595,
        "mean": 11.138927,
        "median": 10.428386,
        "samples": 5
    },
]

means = [v["mean"] for v in data]
mins = [mean - v["min"] for (v, mean) in zip(data, means)]
maxs = [v["max"] - mean for (v, mean) in zip(data, means)]

medians = [v["median"] for v in data]

labels = ["PyTorch CPU", "PyTorch cuda", "burn wgpu"]

ind = np.arange(len(data))

fig, ax = plt.subplots()

ax.grid(axis="y")
ax.set_axisbelow(True)

ax.bar(ind, means, 0.35, yerr=(mins, maxs))
ax.set_xticks(ind)
ax.set_xticklabels(labels=labels)

ax.set_ylabel("секунд")
ax.set_title("Время обучения модели CNN на датасете MNIST (Меньше - лучше)")

plt.savefig("learning_ru.png")