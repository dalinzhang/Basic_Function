import scipy.stats as stats
import numpy as np

ng_measure = [0.8111, 0.7631, 0.6554, 0.7612, 0.7611, 0.7201, 0.7373, 0.7664, 0.7027]
dg_measure = [0.8138, 0.7785, 0.6624, 0.7564, 0.7634, 0.7237, 0.7381, 0.7593, 0.7282]

rank, pVal = stats.wilcoxon(x=dg_measure, y=ng_measure)

print(pVal)
