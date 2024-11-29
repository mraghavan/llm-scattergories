# Overview
The code in this repository simulates and analyzes games of Scattergories using LLMs. There are two main components: data generation and data analysis.

## Data Generation
Data generation proceeds in two steps:
1. Generating samples of LLM responses to Scattergories prompts.
2. Validing answers using an LLM.

Sample generation is done through `generate_samples.py`. Example usage:
```
python3 generate_samples.py -m [MODELS] -s [NUM_SAMPLES] -n [NUM_INSTANCES]
```
This should be run on a GPU cluster. See ```run_generate_samples.slurm``` for an example of how to do this.
Data will be written to `./samples/`.

Sample verification is done through ```verify_samples.py```. Example usage:
```
python3 verify_samples.py -m [MODELS] -v [VERIFIER]
```
This should also be run on a GPU cluster.
See ```run_verify_samples.slurm``` for an example of how to do this.
Data will be written to `./samples/`.
The script `run_all.sh` will schedule both of these `slurm` scripts to run in sequence.
Particularly when testing, consider reducing the number of instances and samples generated.

## Data Analysis
Data analysis has several components:
1. Analyzing equilibria for each langauge model.
2. Analyzing equilibria in pairwise competitions between language models.

For each model, analyzing equilibria is fairly computationally light and need not be done on a cluster. Example usage:
```
python3 analyze_samples.py -m [MODELS] -v [VERIFIER]
```
Data will be written to `./info/`.
Change parameters in the script to analyze different games (e.g., with different congestion functions or numbers of players).

Pairwise equilibria take longer to compute, and should probably be run on a CPU cluster. Example usage:
```
python3 analyze_pairwise.py -m [MODELS] -v [VERIFIER]
```
Data will be written to `./info/`.
See ```run_analyze_pairwise.slurm``` for an example of how to do this on a cluster.

## Plots
This repository also has code to make plots of the results. See `paper_cmds.sh` for the commands used to produce the figures in the paper.

## Models
To add models beyond the ones used in the paper, modify:
1. ```completion_hf.py``` to add the model to the list of models along with its Hugging Face model ID.
2. ```scat_utils.py``` to add the max temperature used for that model.

If running on a Mac instead of a cluster, you can instead add models to ```completion_mlx.py``` and use the ```-x``` flag for ```generate_samples.py```. Make sure these models are MLX models. See ```completion_mlx.py``` for an examples.

The prompt templates require models to support ```system```, ```assistant```, and ```user``` conversation roles.
