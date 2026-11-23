#Outputs:
#results/sweeps/<run_name>/<profile>/<algorithm>/<param(s)>/ 
#results/sweeps/<run_name>/tuned_values.csv
#config/<profile>.yaml                               
#results/<profile>/

import subprocess
import os
import pandas as pd
import yaml


###SETTINGS###


with open("run_all_config.yaml") as f:
    settings = yaml.safe_load(f)


###HELPER FUNCTIONS###


def write_value(config_file, key, value):
    #write one value into a config file by changing only that line
    with open(config_file) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].strip().startswith(key + ":"):
            indent = lines[i][:lines[i].index(key)]
            lines[i] = f"{indent}{key}: {value}  #tuned by run_all.py ({settings['run_name']})\n"#mentioning when what was changed by the sweep
    with open(config_file, "w") as f:
        f.writelines(lines)

#for choosing the best value or group of values from a sweep
def best_row(summary_file, params, metric):
    summary = pd.read_csv(summary_file)
    best_rows = summary[summary[metric + "_mean"] == summary[metric + "_mean"].max()]

    #sort the tied best F1 parameters in the order of the parameter with most swept values, increasing , then do lower median
    #stable to save order
    most_values = list(params)[0]
    for name in params:
        if len(params[name]) > len(params[most_values]):
            most_values = name
    best_rows = best_rows.sort_values(most_values, kind="stable")
    return best_rows.iloc[(len(best_rows) - 1) // 2]  #lower median


###PER PROFILE: SWEEPS, CONFIG, FINAL RUN###

#one row per swept parameter which value was chosen, with its F1/F2, then saved in tuned_values.csv)
chosen_rows = []

#Per SNR profile: Sweep followed by final runs of the pipeline with the chosen values
for profile in settings["profiles"]:

    #SWEEPS: per algorithm
    for algorithm in settings["sweeps"]:
        steps = settings["sweeps"][algorithm]["steps"]
        fixed = dict(settings["sweeps"][algorithm]["start_overrides"])  #values fixed so far for this algorithm

        #sweeps per step (every step has two variables: stage, detection or selection, and parameters to be swept over
        for step in steps:
            stage = step["stage"] #detection or selection
            params = step["params"]  #params is all the values to be swept over, sweep.smk handles differentiating between joint and individual (itertools.product)

            step_name = "__".join(params)  #something like fraction_difference, or neighbourhood_size_max__fraction_difference
            results_dir = f"results/sweeps/{settings['run_name']}/{profile}/{algorithm}/{step_name}"
            summary_file = f"results/sweeps/{settings['run_name']}/{profile}/{algorithm}/{step_name}/sweep_summary.csv"

            #Run the sweep if it hasn't been done yet
            if not os.path.exists(summary_file):
                #settings file for this one sweep, read by sweep/sweep.smk
                #separate from handtuned_sweep_config, that one is for manual tuning. These are saved in the results folder
                sweep_settings = {
                    "profile": profile,
                    "repeats": settings["tuning_repeats"],
                    "algorithm": algorithm,
                    "stage": stage,
                    "params": params,
                    "overrides": fixed,
                    "tuning_data_dir": f"data_simulation/tuning/{settings['run_name']}/{profile}",
                    "results_dir": results_dir,
                }
                
                # save them as <sweep folder>/sweep_settings.yaml
                os.makedirs(results_dir, exist_ok=True)
                sweep_settings_file = f"{results_dir}/sweep_settings.yaml"
                with open(sweep_settings_file, "w") as f:
                    yaml.safe_dump(sweep_settings, f, sort_keys=False)
                
                #Run the sweep, using the sweep_settings.yaml as config
                subprocess.run([
                    "snakemake",
                    "-s", "sweep/sweep.smk",
                    "--workflow-profile", "profiles/sweep",
                    "--config", f"sweep_config={sweep_settings_file}",
                    "--retries", "2",  #Retry a failed job twice, a DLL load error appeared and retrying solved it
                ], check=True)

            if stage == "detection":
                metric = "f2_raw"  #F2 for detection: recall weighs more for detection
            if stage == "selection":
                metric = "f1_sel"

            best = best_row(summary_file, params, metric)
            for param in params:  #so that the multiple parameters for a joint sweep are áll fixed

                value = float(best[param])
                if value == int(value):  #whole numbers as integers so 10, not 10.0
                    value = int(value)
                fixed[param] = value

                chosen_rows.append({
                    "profile": profile,
                    "algorithm": algorithm,
                    "stage": stage,
                    "step": step_name,
                    "param": param,
                    "value": fixed[param],
                    "metric": metric,
                    "mean": best[metric + "_mean"],
                    "ci_low": best[metric + "_ci_low"],
                    "ci_high": best[metric + "_ci_high"],
                })
        
            pd.DataFrame(chosen_rows).to_csv(f"results/sweeps/{settings['run_name']}/tuned_values.csv", index=False)

        #Done with all sweeps for this algorithm, now write the chosen values into the config file for the final run of the pipeline
        config_file = f"config/{profile}.yaml"
        for key in fixed:
            write_value(config_file, key, fixed[key])

        #loop for all algs

    #FINAL RUN: the normal pipeline on the 20 benchmark videos, with the tuned values
    write_value(config_file, "repeats", settings["final_repeats"])

    subprocess.run([
        "snakemake",
        "--config", f"profile={profile}",
        "--forcerun", "run_algorithm", 
        "--retries", "2",               #retry a crashed job twice here too for the DLL load error
    ], check=True)                      #check=True: stop run_all.py if snakemake fails