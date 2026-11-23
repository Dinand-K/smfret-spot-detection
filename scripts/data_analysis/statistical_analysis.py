import pandas as pd
import numpy as np
from scipy import stats


###LOAD VARIABLES###


input_class_files  = snakemake.input.class_benchmarks
input_global_files = snakemake.input.global_benchmarks
# Outputs
output_class_stats  = snakemake.output.stats_class
output_global_stats = snakemake.output.stats_global


###HELPER FUNCTIONS###


#95% confidence interval of the mean over repeats
#values is one value per reruns of the simulation (in this case the F1 scores, recalls or precisions etc)
def ci_half_width(values):
    n_repeats = values.count()

    if n_repeats == 0: 
        raise ValueError("No values to compute a CI from: every metric should have one value per repeat")
    if n_repeats == 1:  #Real Dataset ór test run with a single repeat
        return np.nan
    
    standard_deviation = values.std()                             #sample standard deviation (degrees of freedom automatic value is n-1 because repeats are sample of all possible sims) over the repeats
    standard_error_of_mean = standard_deviation / np.sqrt(n_repeats)
    t_critical_value = stats.t.ppf(0.975, n_repeats - 1)          #t critical value for 95% (2.5% in each tail)
    return t_critical_value * standard_error_of_mean

# helperfunctions
def ci_low(values): #Lower bound of the 95% CI of the mean
    return values.mean() - ci_half_width(values)

def ci_high(values): #upper bound of the 95% CI of the mean
    return values.mean() + ci_half_width(values)

statistics_per_metric = ['mean', ci_low, ci_high]


###BENCHMARKING###


#class benchmarks: stack the tables of all repeats, so every class has one row per repeat
class_tables = []
for class_file in input_class_files:
    class_tables.append(pd.read_csv(class_file, index_col="Class"))
class_all_repeats = pd.concat(class_tables)
#mean and 95% CI per class over the repeats
class_statistics = class_all_repeats.groupby("Class").agg(statistics_per_metric)

#global benchmarks (precision, recall, F1): same, with one row per metric per repeat
global_tables = []
for global_file in input_global_files:
    global_tables.append(pd.read_csv(global_file, index_col=0))
global_all_repeats = pd.concat(global_tables)
global_all_repeats.index.name = "Metric" #first column has no name in the csv
global_statistics = global_all_repeats.groupby("Metric").agg(statistics_per_metric)


###SAVING###

class_statistics.to_csv(output_class_stats)
global_statistics.to_csv(output_global_stats)

