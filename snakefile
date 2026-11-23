# TO Use manually: snakemake --config profile=snr709  (Profile is snr709 or snr2 or snr05 or snr03 or realdata)

PROFILE = config["profile"]
configpath = f"config/{PROFILE}.yaml"
configfile: configpath

#Outputs
RES = f"results/{PROFILE}"
SIM = f"data_simulation/{PROFILE}"

REPEATS = config["repeats"]
MATCHING_THRESHOLD = config["matching_threshold"]

ALGORITHMS = []
for alg_name in config["algorithms"]:
    if config["algorithms"][alg_name]["activate"]:
        ALGORITHMS.append(alg_name)


REALDATA = config["use_real_data"]

### RULES ###
rule all:
    input:
        expand(RES + "/{alg}/statistics_global.csv", alg=ALGORITHMS),
        expand(RES + "/{alg}/statistics_class.csv", alg=ALGORITHMS),

rule generate_data: #(though also used when using real data! The data_simulation script has a realdata pathway inside)
    output:
        tiff    = SIM + "/tifffile/rep{rep}.tiff",
        coords  = SIM + "/particle_coordinates/rep{rep}.npy",
        classes = SIM + "/particle_classes/rep{rep}.npy"
    params:
        sim_settings = config["simulation"],
        use_real_data = REALDATA  #switches between data sim and simply reading provided files inside data_simulation.py
    script:
        "scripts/data_analysis/data_simulation.py"


rule run_algorithm:
    input:
        SIM + "/tifffile/rep{rep}.tiff"
    output:
        det = RES + "/{alg}/detections/rep{rep}.npy",
        sel = RES + "/{alg}/selections/rep{rep}.npy"
    conda:
        lambda wc: f"envs/{wc.alg}.yaml"
    params:
        script_path = lambda wc: f"scripts/algorithm_wrappers/{wc.alg}_wrapper.py",
        config_path = configpath
    threads:
        lambda wc: config["algorithms"][wc.alg]["threads"]
    resources:
        serial_lock = lambda wc: config["algorithms"][wc.alg]["serial_lock_cost"]
    shell:
        "python {params.script_path} {input} {output.det} {output.sel} {params.config_path}"

rule matching_benchmarking:
    input:
        detected_points   = RES + "/{alg}/detections/rep{rep}.npy",
        classified_points = RES + "/{alg}/selections/rep{rep}.npy",
        groundtruth_detections      = SIM + "/particle_coordinates/rep{rep}.npy",
        groundtruth_classifications = SIM + "/particle_classes/rep{rep}.npy"
    output:
        class_benchmarks  = RES + "/{alg}/class_benchmarks/rep{rep}.csv",
        global_benchmarks = RES + "/{alg}/global_benchmarks/rep{rep}.csv"
    params:
        matching_threshold = MATCHING_THRESHOLD,
        sim_settings = config["simulation"]
    script:
        "scripts/data_analysis/matching_and_benchmarking.py"

rule statistics:
    input: #Expand needs double brackets, single would mean it tries to fill in alg, so this way alg stays wildcard
        class_benchmarks  = expand(RES + "/{{alg}}/class_benchmarks/rep{rep}.csv", rep=range(1, REPEATS+1)),
        global_benchmarks = expand(RES + "/{{alg}}/global_benchmarks/rep{rep}.csv", rep=range(1, REPEATS+1))
    output:
        stats_class  = RES + "/{alg}/statistics_class.csv",
        stats_global = RES + "/{alg}/statistics_global.csv"
    script:
        "scripts/data_analysis/statistical_analysis.py"
