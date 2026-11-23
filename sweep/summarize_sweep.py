#Summary of one sweep: per swept value the mean and 95% CI over the tuning videos of the pipeline's detection and selection precision, recall and F1, and the detection F2
#Saved to results/sweeps/(run_name)/(profile)/(algorithm)/(sweep step)>/sweep_summary.csv
import numpy as np
import pandas as pd
from scipy import stats


###SETTINGS###


sweep_settings = snakemake.params.sweep_settings   #either handtuned_sweep_config.yaml or written by run_all.py
combinations = snakemake.params.combinations  

swept_parameters = list(sweep_settings["params"])
n_videos = sweep_settings["repeats"]
t_critical_value = stats.t.ppf(0.975, n_videos - 1)


###MEAN AND 95% CI PER COMBINATION###


summary_rows = []
for folder_name in combinations:

    #the metrics of every tuning video for this combination frmo the main pipeline's global benchmarks
    # and per metric a list with one value per tuning video
    values = {
        "precision_raw": [],
        "recall_raw":    [],
        "f1_raw":        [],
        "precision_sel": [],
        "recall_sel":    [],
        "f1_sel":        [],
        "f2_raw":        [],
    }

    for rep in range(1, sweep_settings["repeats"] + 1):
        #get global benchmarks per tuning video for a swept variable
        benchmark_file = f"{sweep_settings['results_dir']}/{folder_name}/global_benchmarks/rep{rep}.csv"
        global_benchmarks = pd.read_csv(benchmark_file, index_col=0)  #rows Global_Precision/Recall/F1_Score, columns Raw (detection) and Filtered (selection)

        #extract the metrics of interest
        detection_precision = global_benchmarks.loc["Global_Precision", "Raw"]
        detection_recall = global_benchmarks.loc["Global_Recall", "Raw"]
        values["precision_raw"].append(detection_precision)
        values["recall_raw"].append(detection_recall)
        values["f1_raw"].append(global_benchmarks.loc["Global_F1_Score", "Raw"])
        values["precision_sel"].append(global_benchmarks.loc["Global_Precision", "Filtered"])
        values["recall_sel"].append(global_benchmarks.loc["Global_Recall", "Filtered"])
        values["f1_sel"].append(global_benchmarks.loc["Global_F1_Score", "Filtered"])

        #also calculate detection F2
        numerator = 5 * detection_precision * detection_recall
        denominator = 4 * detection_precision + detection_recall
        if denominator > 0:
            values["f2_raw"].append(numerator / denominator)
        else:
            values["f2_raw"].append(0.0)  #recall and precison are zero just means F2=0

    #gets the parameter and its value for this combination
    summary_row = combinations[folder_name]
    #using the values lists to make the summary row with mean and 95% CI of the mean for each metric
    for metric in values:
        metric_values = np.array(values[metric])

        #statistics
        mean = np.mean(metric_values)
        standard_error_of_mean = np.std(metric_values, ddof=1) / np.sqrt(n_videos)  #ddof=1: sample standard deviation
        ci_half_width = t_critical_value * standard_error_of_mean

        summary_row[metric + "_mean"]    = mean
        summary_row[metric + "_ci_low"]  = mean - ci_half_width
        summary_row[metric + "_ci_high"] = mean + ci_half_width

    #add this swept variable's summary to the summary CSV
    summary_rows.append(summary_row)


###SAVING###

#sorted by the first swept parameter, then the next, because run_all.py relies on this order to break ties in joint sweeps
sweep_summary = pd.DataFrame(summary_rows).sort_values(swept_parameters, kind="stable")
sweep_summary.to_csv(snakemake.output.csv, index=False)
