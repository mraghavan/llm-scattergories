# Overview
The code in this repository implements [this paper](https://arxiv.org/abs/2412.08610).
At a high level, it simulates and analyzes games of Scattergories using LLMs. There are two main components: data generation and data analysis.

## Data Generation
Data generation proceeds in two steps:
1. Generating samples of LLM responses to Scattergories prompts.
2. Validing answers using an LLM.

Sample generation is done through [`generate_samples.py`](generate_samples.py). Example usage:
```
python3 generate_samples.py -m [MODELS] -s [NUM_SAMPLES] -n [NUM_INSTANCES]
```
This should be run on a GPU cluster. See [`run_generate_samples.slurm`](run_generate_samples.slurm) for an example of how to do this.
Data will be written to `./samples/`.

Sample verification is done through [`verify_samples.py`](verify_samples.py). Example usage:
```
python3 verify_samples.py -m [MODELS] -v [VERIFIER]
```
This should also be run on a GPU cluster.
See [`run_verify_samples.slurm`](run_verify_samples.slurm) for an example of how to do this.
Data will be written to `./samples/`.
The script [`run_all.sh`](run_all.sh) will schedule both of these `slurm` scripts to run in sequence.
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
See [`run_analyze_pairwise.slurm`](run_analyze_pairwise.slurm) for an example of how to do this on a cluster.

## Plots
This repository has code to make all of the plots in the paper. Below is a list of the plotting scripts and the output files they generate.

*   `python3 plot_s.py`: Generates `img/s_gamma.png` and `img/x_over_s_gamma.png`.
*   `python3 plot_inf.py`: Generates `img/lim_opt_3d.png` and `img/lim_eq_3d.png`.
*   `python3 plot_opt_over_temp.py -m [MODELS] -v [VERIFIER]`: Generates `[model]_opt_over_temp.png` for each model.
*   `python3 make_plots.py -m [MODELS]`: Generates aggregated equilibrium plots:
    *   `opt_and_eq_util_over_gamma.png`
    *   `opt_and_eq_temp_over_gamma.png`
    *   `opt_and_eq_sw_over_gamma.png`
    *   `opt_and_eq_util_over_n.png`
    *   `opt_and_eq_temp_over_n.png`
*   `python3 plot_pairwise.py -m [MODEL1],[MODEL2]`: Generates pairwise equilibrium plots:
    *   `[model1]_[model2]_gamma_[gamma]_pairwise_counts_scatter.png`
    *   `[model1]_[model2]_gamma_[gamma]_pairwise_temp.png`
    *   `[model1]_[model2]_gamma_[gamma]_pairwise_utility.png`
*   `python3 compare_rankings.py -n [NUM_INSTANCES] --min-count [MIN_COUNT]`: Generates spectral analysis plots (see "Spectral analysis" below for prerequisites):
    *   `img/distance_matrix.png`
    *   `img/spectral_embedding_2d.png`
    *   `img/spectral_embedding_3d.png`
    *   `img/confusion_matrix.png`
*   `python3 make_big_pairwise_plot.py -m [MODELS]`: Generates `all_market_share_pairwise.png`.
*   `python3 plot_inversions.py -m [MODELS]`: Generates `weighted_inversions.png` and `mass_captured.png`.
*   `python3 plot_prs.py -m [MODELS] -v [VERIFIER]`: Generates `pr_correct.png`.

See [`paper_cmds.sh`](paper_cmds.sh) for the exact commands used to produce the figures in the paper.

## Models
To add models beyond the ones used in the paper, modify:
1. [`completion_hf.py`](completion_hf.py) to add the model to the list of models along with its Hugging Face model ID.
2. [`scat_utils.py`](scat_utils.py) to add the max temperature used for that model.

If running on a Mac instead of a cluster, you can instead add models to [`completion_mlx.py`](completion_mlx.py) and use the ```-x``` flag for ```generate_samples.py```. Make sure these models are MLX models. See ```completion_mlx.py``` for an examples.

The prompt templates require models to support ```system```, ```assistant```, and ```user``` conversation roles.

# Spectral analysis
To reproduce the spectral analysis plots (via `compare_rankings.py`), you must generate the data for the spectral analysis.
Start with
```
python3 make_model_configs.py --max-temp-only
```
to generate a config file for each (LLM, prompt) pair in ```./models```.
Then, run
```
sbatch run_generate_all_answers.slurm
```
to generate 50 samples per model config per Scattergories instance. Adjust the ```-n [num_instances]``` flag in the file as necessary.

Next, run
```
sbatch run_verify_answers.slurm
```
to verify answer correctness.

Then, run
```
python3 candidate_answers.py --min-count [min_count] -n [num_instances]
```
to take only correct answers that appear ```[min_count]``` times across all of the sampes. Again, adjust the ```-n [num_instances]``` flag. This will write to ```./answer_sets```.
Then, run
```
python3 make_model_configs.py
```
to generate model configs for (LLM, prompt, temperature) tuples.

Next, run
```
sbatch run_evaluate_likelihoods.slurm [min_count]
```
to evaluate the likelihood of each answer for each (LLM, prompt, temperature) tuple. Again, adjust ```-n [num_instances]``` as necessary, and make sure ```[min_count]``` matches what you used in the previous step. This will write to ```./rankings```.

Finally, run
```
python3 compare_rankings.py -n [num_instances] --min-count [min_count]
```
to analyze the data and produce the plots. The paper uses ```num_instances=10``` and ```min_count=10```.