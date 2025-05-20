# Rooftop PV Panel Clustering

This repository implements the method proposed in the paper [Clustering Rooftop PV Systems via Probabilistic Embeddings (Bölat et al. 2023)](https://arxiv.org/abs/2505.10699) for clustering rooftop photovoltaic (PV) generation profiles. We provide example data and scripts to reproduce the analysis and/or apply the method to your own data. Full data (hourly 15 min resolution) for ∼175 PV systems over 1461 days can be obtained from https://zenodo.org/records/6906504.

## 🔢 Example Data Format

We include a small example in `data/X_daily_15min_example.csv` spanning 4 years for 2 PV systems. The columns `X_0` … `X_95` correspond to the 96 15 min slices over one day, repeated for each day in sequence. The `DATE` and `ID` columns indicate the date and system ID. Metadata in `data/metadata.csv` provides capacities, tilt, azimuth, etc.

```csv
X_0,X_1,…,X_95,DATE,ID
0.0,0.0,…,0.0,2014-01-01,1
0.0,0.0129,…,0.4931,2014-01-02,1
…
```


## 🖥️ Usage

### Jupyter Notebook

Open `pv_clustering.ipynb` to:
- Visualize raw, scaled, and log-normalized profiles
- Compute data availability
- Apply dimensionality reduction & clustering
- Plot cluster centers and time series

### Scripts

- `collect_results_vanilla.py`: evaluates the dispersion scores of different clustering methods on full dataset and saves results to `results/`
- `collect_results_loo.py`: evaluates the sensitivity of clustering methods to the removal of individual systems (leave-one-out) and saves results to `results/`

```bash
python collect_results_vanilla.py
python collect_results_loo.py
```

## 📈 Results

Generated figures and CSVs are in `results/`:
- Figures: `ablation.png`, `clusters_and_methods.png`, `quantile_representation.png`
- CSVs: `results_vanilla.csv`, `results_loo.csv`

## 🔗 Dependencies

- Python 3.7+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- fastnanquantile
- tqdm

Install via:

```bash
pip install -r requirements.txt
```



## 📄 Citation
If you use this code or data in your research, please cite the following paper:

```
@misc{bölat2025clusteringrooftoppvsystems,
      title={Clustering Rooftop PV Systems via Probabilistic Embeddings}, 
      author={Kutay Bölat and Tarek Alskaif and Peter Palensky and Simon Tindemans},
      year={2025},
      eprint={2505.10699},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.10699}, 
}
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgements
This software is developed under the H2020-MSCA-ITN [Innovative Tools for Cyber-Physical Systems (InnoCyPES)](https://innocypes.eu/) project. The project is funded by the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 956433.

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" alt="drawing" width="150"/> 