import pandas as pd
import seaborn as sns

data = [
    ["GPU", 10, 0.27160120010375977],
    ["GPU", 2500, 0.3027505874633789],
    ["GPU", 5000, 0.5957744121551514],
    ["GPU", 7500, 1.1456775665283203],
    ["GPU", 10000, 1.7748563289642334],
    ["GPU", 12500, 2.648530960083008],
    ["GPU", 15000, 4.285149812698364],
    ["CPU", 10, 0.3386576175689697],
    ["CPU", 2500, 2.1455774307250977],
    ["CPU", 5000, 8.316681385040283],
    ["CPU", 7500, 18.45176863670349],
    ["CPU", 10000, 33.32748246192932],
    ["CPU", 12500, 52.00561046600342],
    ["CPU", 15000, 74.90831971168518]
]
df = pd.DataFrame(data, columns=["Divice", "Particles", "Time (s)"])
sns.set_theme()
plot = sns.relplot(data=df, x="Particles", y="Time (s)", hue="Divice")
plot.fig.savefig("time.jpg")