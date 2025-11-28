import pandas as pd
import gutenTAG.api as gt
import matplotlib.pyplot as plt

N = 500
ts_sine = gt.sine(length=N, frequency=2, amplitude=1.8)
ts_dirichlet = gt.dirichlet(length=N)

# anomaly injection TBD

df = pd.DataFrame({"ch-0": ts_sine, "ch-1": ts_dirichlet})

df.plot()
plt.show()