#Run this:
#snakemake -s sweep/matching_threshold_snr03.smk --workflow-profile profiles/sweep

# This only re-runs performance analysis for a given set of matching distances, algorithms, repeats and SNR=0.3. 
# Does not redo actual data simulation and algorithm runs

#Variables
THRESHOLDS = ["0.5", "1", "2", "3"]   #px (text, because they are also part of the folder names)
ALGORITHMS = ["papylio", "stardist"]
REPEATS = 20                          #videos in the last run, must be at most the amount of detection and selection point sets in the results folder

#Selecting sim and results of profile I want to sweep matching distances over:
SIM = "data_simulation/snr03"         
RUN = "results/snr03" 
configfile: "config/snr03.yaml"

#Output: new matchingdistance_sweep results folder
RES = "results/matchingdistance_sweep/matching_threshold_{threshold}"

rule all:
    input:
        expand(RES + "/{alg}/statistics_global.csv", threshold=THRESHOLDS, alg=ALGORITHMS)

#same as main except treshold
rule matching_benchmarking:
    input:
        detected_points   = RUN + "/{alg}/detections/rep{rep}.npy",
        classified_points = RUN + "/{alg}/selections/rep{rep}.npy",
        groundtruth_detections      = SIM + "/particle_coordinates/rep{rep}.npy",
        groundtruth_classifications = SIM + "/particle_classes/rep{rep}.npy"
    output:
        class_benchmarks  = RES + "/{alg}/class_benchmarks/rep{rep}.csv",
        global_benchmarks = RES + "/{alg}/global_benchmarks/rep{rep}.csv"
    params:
        matching_threshold = lambda wc: float(wc.threshold),
        sim_settings = config["simulation"]
    script:
        "../scripts/data_analysis/matching_and_benchmarking.py"

#same as main
rule statistics:
    input:
        class_benchmarks  = expand("results/matchingdistance_sweep/matching_threshold_{{threshold}}/{{alg}}/class_benchmarks/rep{rep}.csv", rep=range(1, REPEATS+1)),
        global_benchmarks = expand("results/matchingdistance_sweep/matching_threshold_{{threshold}}/{{alg}}/global_benchmarks/rep{rep}.csv", rep=range(1, REPEATS+1))
    output:
        stats_class  = RES + "/{alg}/statistics_class.csv",
        stats_global = RES + "/{alg}/statistics_global.csv"
    script:
        "../scripts/data_analysis/statistical_analysis.py"
