# smfret-spot-detection

Included in this github page are the code and results files for the preprint:  
*A pre-trained deep-learning segmentation model repurposed for single-molecule FRET spot detection* (\[bioRxiv link, TODO]).  
It benchmarks StarDist (model: `2D_versatile_fluo`, not retrained) against Papylio v0.9 in finding and selecting for target single-molecule events in simulated TIRF smFRET videos

A Snakemake pipeline (snakefile) simulates videos at four noise levels, or runs with hand-annotated experimental video data (realdata/), then runs both algorithms and finally scores them against the ground truth.

## Installation

The code was made on Windows 11 and also works on Linux. Conda is required. It can be installed by running:

```
git clone https://github.com/<user>/smfret-spot-detection.git
cd smfret-spot-detection
conda env create -f environment.yml
conda activate snakemake
```

This installs only Snakemake and the packages the analysis scripts need. Snakemake builds the StarDist and Papylio environments from envs/ on its first run.

## Running

Run from the project folder in the `snakemake` environment. Running a profile overwrites its folder in `results/`.

Simulate videos and run both algorithms at one noise level: profile = snr709, snr2, snr05 or snr03.
Each of their settings (how many sims, which algorithms are included, their settings etc.) are found in config/

```
snakemake --config profile=snr709
```

Run with real data, with vid.tiff, coord.npy and class.npy from 4TU in realdata/.
Settings can be found in config/

```
snakemake --config profile=realdata
```

Run whole experiment: tuning sweeps + final runs.
Uses settings from all config/ files, but tunes parameters specified in run\_all config.
Before running, set a new run\_name in run\_all\_config.yaml, otherwise the published sweeps are reused instead of rerun.

```
python run_all.py
```

Run the matching-distance sweep (Figure 5).
This redoes the SNR03 results statistical analysis (matching detections to ground-truth), so SNR03 results are required.

```
snakemake -s sweep/matching_threshold_snr03.smk --workflow-profile profiles/sweep
```

## Acknowledgements

Developed in the Joo Lab at TU Delft under the supervision of Prof. Dr. C. Joo and Dr. C. de Lannoy.

## References

StarDist  
Schmidt, U., Weigert, M., Broaddus, C., \& Myers, G. (2018). Cell detection with star-convex polygons. In A. F. Frangi, J. A. Schnabel, C. Davatzikos, C. Alberola-López, \& G. Fichtinger (Eds.), *Medical Image Computing and Computer Assisted Intervention – MICCAI 2018* (Lecture Notes in Computer Science, Vol. 11071, pp. 265–273). Springer. https://arxiv.org/abs/1806.03535

Papylio  
Severins, I., Kerssemakers, J., Kim, S. H., Bastiaanssen, C., Docter, M., \& Joo, C. (2025). *Papylio* (Version 0.9) \[Computer software]. GitHub. https://github.com/Chirlmin-Joo-lab/papylio/blob/a2bec3bc1d736c5427fb3d0e804d45a66bd4cfde/CITATION.cff

Snakemake: workflow manager  
Mölder, F., Jablonski, K. P., Letcher, B., Hall, M. B., Tomkins-Tinch, C. H., Sochat, V., Forster, J., Lee, S., Twardziok, S. O., Kanitz, A., Wilm, A., Holtgrewe, M., Rahmann, S., Nahnsen, S., \& Köster, J. (2021). Sustainable data analysis with Snakemake. *F1000Research*, 10, 33. https://doi.org/10.12688/f1000research.29032.2

## Reproducibility

Exact package versions are pinned in environment.yml and envs/

## Data, citation and licence

* **Experimental video:** `vid.tiff` can be found on: \[TODO].
* **Citation:** if you use this code, please cite the preprint: \[TODO].
* **Licence:** MIT, see `LICENSE` \[TODO].

