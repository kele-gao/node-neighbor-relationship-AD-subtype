# Connectome-based subtype analysis

Analysis workflow: ComBat harmonization → W-score calculation → connectome-based subtype clustering.

## Setup

Requires Python 3 and the following packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib statsmodels neuroCombat
```

Set `MAIN_DIR` in each analysis script to your local code directory. Record package versions used for the manuscript analysis.

## Input data

The current data contain 50 simulated subjects, not real participant data. These simulated data are provided only to demonstrate the analysis workflow and cannot reproduce the manuscript results. The `output/` directory contains example analysis results generated from the simulated data. 

Place nine ROI-by-ROI connectome matrices in `data/networks/`

The connectome matrices could be obtained from Hansen, J. Y. *et al.* Integrating multimodal and multiscale connectivity blueprints of the human cerebral cortex in health and disease. *PLoS Biol* **21**, e3002314 (2023). Source repository: [hansen_many_networks](https://github.com/netneurolab/hansen_many_networks).

## Run

From the `code` directory, run in order:

pip install -r requirements.txt
python scripts/C1_combat.py
python scripts/C2_Wscore.py
python scripts/C3_cluster_subtype.py

Later steps read the preceding step's outputs automatically.

## License

Original code in this repository is licensed under the [MIT License](LICENSE). Third-party software, datasets, and connectome matrices retain their original licenses and terms of use; the MIT License does not relicense those materials.
