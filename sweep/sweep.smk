# Parameter sweep: run one algorithm on tuning videos for every value of the swept parameter(s) and summarise the
# results per value. Two or more parameters = joint sweep: every combination of their values.

#Run by hand:
#   snakemake -s sweep/sweep.smk --workflow-profile profiles/sweep --config sweep_config=sweep/handtuned_sweep_config.yaml

#running by hand means using handtuned_sweep_config.yaml !
#run_all.py does the same with its own settings yaml, each put into the respective results folders, so using run_all does not use the handtuned_sweep_config.yaml

import itertools
import yaml


###SETTINGS###


#the settings of this sweep
with open(config["sweep_config"]) as file:
    SWEEP_SETTINGS = yaml.safe_load(file)
ALGORITHM = SWEEP_SETTINGS["algorithm"]
SIM = SWEEP_SETTINGS["tuning_data_dir"]   #folder with the tuning videos like main snakefile
RES = SWEEP_SETTINGS["results_dir"]       #folder for the sweep results like main snakefile

#the SNR profile config the sweep uses a basis to sweep the variables over
with open(f"config/{SWEEP_SETTINGS['profile']}.yaml") as f:
    PROFILE_CONFIG = yaml.safe_load(f)
PROFILE_CONFIG["simulation"]["global_settings"]["seed"] += 100000  #tuning videos get diff seeds from benchmarking videos


#every combination of swept values gets its own folder named after the parameters and their values
def make_folder_name(combination):
    parts = []
    for name in combination:
        parts.append(f"{name}_{combination[name]:g}")
    return "__".join(parts)

COMBINATIONS = {}  #folder name is values
for values in itertools.product(*SWEEP_SETTINGS["params"].values()):  # all combinations of the value lists
    combination = dict(zip(SWEEP_SETTINGS["params"], values))         # pair each parameter name with its value
    COMBINATIONS[make_folder_name(combination)] = combination



###RULES###



rule all:
    input:
        RES + "/sweep_summary.csv"

rule generate_data:
    output:
        tiff    = SIM + "/tifffile/rep{rep}.tiff",
        coords  = SIM + "/particle_coordinates/rep{rep}.npy",
        classes = SIM + "/particle_classes/rep{rep}.npy"
    params:
        sim_settings = PROFILE_CONFIG["simulation"],
        use_real_data = False
    script:
        "../scripts/data_analysis/data_simulation.py"

#the profile config with the overrides and then the swept value (or values for joint sweep) filled in
rule make_sweep_config:
    output:
        RES + "/{combination}/config.yaml"
    params:
        overrides = SWEEP_SETTINGS["overrides"]  #as params, so Snakemake redoes this config when the overrides change
    run:
        #load config dic
        with open(f"config/{SWEEP_SETTINGS['profile']}.yaml") as f:
            combination_config = yaml.safe_load(f)
        algorithm_params = combination_config["algorithms"][ALGORITHM]["params"]

        #fill in override variables
        for key in params.overrides:
            algorithm_params[key] = params.overrides[key]

        #use swept variable
        swept_values = COMBINATIONS[wildcards.combination]  # wildcards.combination = this folder's name
        for key in swept_values:  #after the overrides, so the swept values win
            algorithm_params[key] = swept_values[key]

        #save as this folders config
        with open(output[0], "w") as f:
            yaml.safe_dump(combination_config, f, sort_keys=False)

#run normal wrapper, same as usual snakefile
rule run_algorithm:
    input:
        tiff        = SIM + "/tifffile/rep{rep}.tiff",
        config_file = RES + "/{combination}/config.yaml"
    output:
        det = RES + "/{combination}/detections/rep{rep}.npy",
        sel = RES + "/{combination}/selections/rep{rep}.npy"
    conda:
        f"../envs/{ALGORITHM}.yaml"
    threads:
        PROFILE_CONFIG["algorithms"][ALGORITHM]["threads"]
    resources:
        serial_lock = PROFILE_CONFIG["algorithms"][ALGORITHM]["serial_lock_cost"]  #Papylio shares one working folder, so its runs must not overlap
    shell:
        f"python scripts/algorithm_wrappers/{ALGORITHM}_wrapper.py {{input.tiff}} {{output.det}} {{output.sel}} {{input.config_file}}"

#same as normal too
rule matching:
    input:
        detected_points   = RES + "/{combination}/detections/rep{rep}.npy",
        classified_points = RES + "/{combination}/selections/rep{rep}.npy",
        groundtruth_detections      = SIM + "/particle_coordinates/rep{rep}.npy",
        groundtruth_classifications = SIM + "/particle_classes/rep{rep}.npy"
    output:
        class_benchmarks  = RES + "/{combination}/class_benchmarks/rep{rep}.csv",
        global_benchmarks = RES + "/{combination}/global_benchmarks/rep{rep}.csv"
    params:
        matching_threshold = PROFILE_CONFIG["matching_threshold"],
        sim_settings = PROFILE_CONFIG["simulation"]
    script:
        "../scripts/data_analysis/matching_and_benchmarking.py"

#all combinations x all tuning videos 
rule summarize:
    input:
        expand(RES + "/{combination}/global_benchmarks/rep{rep}.csv", combination=list(COMBINATIONS), rep=range(1, SWEEP_SETTINGS["repeats"] + 1))
    output:
        csv = RES + "/sweep_summary.csv",
    params:
        sweep_settings = SWEEP_SETTINGS,
        combinations = COMBINATIONS
    script:
        "summarize_sweep.py"
