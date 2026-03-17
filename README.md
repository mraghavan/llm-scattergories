# Overview
The code in this repository implements [this paper](https://arxiv.org/abs/2412.08610).
At a high level, it simulates and analyzes games of Scattergories using LLMs.

There are two main workflows:
- the original paper workflow for equilibrium analysis
- the current ranking-and-clustering workflow, which generates samples, verifies them, builds shared candidate answer sets, scores answer likelihoods, and clusters model configurations by their answer rankings

## Data Generation
Data generation for the original paper workflow proceeds in two steps:
1. generating samples of LLM responses to Scattergories prompts
2. validating answers using an LLM

Sample generation is done through [`generate_samples.py`](generate_samples.py). Example usage:
```
python3 generate_samples.py -m [MODELS] -s [NUM_SAMPLES] -n [NUM_INSTANCES]
```
This should be run on a GPU cluster. See [`run_generate_samples.slurm`](run_generate_samples.slurm) for an example of how to do this.
Data will be written to `./out/`.

Sample verification is done through [`verify_samples.py`](verify_samples.py). Example usage:
```
python3 verify_samples.py -m [MODELS] -v [VERIFIER]
```
This should also be run on a GPU cluster.
See [`run_verify_samples.slurm`](run_verify_samples.slurm) for an example of how to do this.
Data will be written to `./out/`.
The script [`run_all.sh`](run_all.sh) will schedule both of these `slurm` scripts to run in sequence.
Particularly when testing, consider reducing the number of instances and samples generated.

## Ranking Pipeline
The current end-to-end ranking pipeline has five stages:
1. generate samples for each `(model, prompt, temperature)` config
2. verify those samples with `qwen2.5`
3. build a shared candidate-answer set for each Scattergories instance
4. evaluate answer likelihoods for every config on that shared answer set
5. compare rankings and run spectral clustering

### Canonical outputs
The pipeline writes to these directories:
- `models/`: one config JSON per `(model, prompt, temperature)`
- `out/`: generated samples and verification results
- `answer_sets/`: shared candidate answer sets per instance
- `rankings/`: per-config answer likelihood rankings
- `info/`: saved clustering artifacts such as `compare_rankings_n25_min10.pkl`

### Config Set
The ranking pipeline in this README generates results for:
- `6` base models
- `14` prompt variants total
- `3` temperatures per model

This gives `252` total configs.

This includes `3` strategic prompts, which are used in the paper appendix.

### One-command cluster run
The simplest way to run the full pipeline is:

```bash
bash submit_models_pipeline.sh 10
```

This submits the full `n=25`, `min_count=10` pipeline in order:
- [`run_generate_all_answers.slurm`](run_generate_all_answers.slurm)
- [`run_verify_answers.slurm`](run_verify_answers.slurm)
- [`run_candidate_answers.slurm`](run_candidate_answers.slurm)
- [`run_evaluate_likelihoods.slurm`](run_evaluate_likelihoods.slurm)
- [`run_compare_rankings.slurm`](run_compare_rankings.slurm)

The argument to `submit_models_pipeline.sh` is `min_count`.

### What each stage runs
1. Generate model configs

```bash
find models -maxdepth 1 -name '*.json' -delete
python3 make_model_configs.py
```

2. Generate samples

```bash
sbatch run_generate_all_answers.slurm
```

Current defaults in [`run_generate_all_answers.slurm`](run_generate_all_answers.slurm):
- `n=25`
- `s=50`
- array job over config shards
- `--enforce_starting_token`

This writes `*_samples.pkl` files to `out/`.

3. Verify answers

```bash
sbatch run_verify_answers.slurm
```

This runs:

```bash
python3 verify_samples.py -c models -v qwen2.5 -i out
```

This writes `*_qwen2.5_verified.pkl` files to `out/`.

4. Build candidate answer sets

```bash
sbatch run_candidate_answers.slurm 10 25
```

This runs:

```bash
python3 candidate_answers.py --min-count 10 -n 25 --input-dir out
```

This writes `*_min10_answers.pkl` files to `answer_sets/`.

5. Evaluate answer likelihoods

```bash
sbatch run_evaluate_likelihoods.slurm 10
```

This runs:

```bash
python3 evaluate_answer_likelihoods.py --job-num $SLURM_ARRAY_TASK_ID --num-jobs $SLURM_ARRAY_TASK_COUNT --min-count 10 -n 25
```

This writes `*_min10_rankings.pkl` files to `rankings/`.

6. Compare rankings and cluster

```bash
python3 compare_rankings.py --no-plots --min-count 10 -n 25 --output-pkl info/compare_rankings_n25_min10.pkl
```

If you want to reproduce the clustering figures from the paper using only the original prompt families, run:

```bash
python3 compare_rankings.py --original-prompts-only --min-count 10 -n 25
```

If you want only the saved paper-style clustering artifact without plots, run:

```bash
python3 compare_rankings.py --no-plots --original-prompts-only --min-count 10 -n 25 --output-pkl info/compare_rankings_n25_min10_original.pkl
```

This restricts clustering to the original 11 prompt families while leaving upstream generation and likelihood-evaluation outputs unchanged.

### Reuse behavior
The pipeline can reuse existing files:
- sample generation skips configs/instances that already have enough samples
- verification reuses existing verification files unless the underlying sample files changed
- likelihood evaluation reuses existing ranking files unless you delete them

If you change sampling behavior in a way that affects the generated distributions, the clean rerun sequence is:
1. delete `out/*_samples.pkl`
2. keep or delete verification files depending on whether they are still valid for the new samples
3. delete `answer_sets/*`
4. delete `rankings/*`
5. delete any stale `info/compare_rankings_*.pkl`
6. rerun the full pipeline

### Coverage checks
For the canonical `n=25` run, the expected file counts are:
- samples: `252 x 25 = 6300`
- rankings: `252 x 25 = 6300`
- answer sets: `25`
- verification files: `25`

When complete, `compare_rankings.py` should produce a pickle like:
- `info/compare_rankings_n25_min10.pkl`

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

# Spectral Analysis
To reproduce the current spectral analysis and clustering workflow, use the ranking pipeline above.

The default saved artifact is:

```bash
info/compare_rankings_n25_min10.pkl
```

If you want to reproduce the clustering figures from the paper, run:

```bash
python3 compare_rankings.py --original-prompts-only -n 25 --min-count 10
```

If you only want the corresponding paper-style data artifact on a cluster without rendering plots, run:

```bash
python3 compare_rankings.py --no-plots --original-prompts-only -n 25 --min-count 10 --output-pkl info/compare_rankings_n25_min10_original.pkl
```

The saved pickle contains:
- `avg_distance_matrix`
- `distance_matrices`
- `labels`
- `embedding_2d`
- `embedding_3d`
- `confusion_matrix`
- `model_ids`
- `model_names`
- `accuracy`
- `similarity_matrix`
- `true_label_names`
- `cluster_names`
