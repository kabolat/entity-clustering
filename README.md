# Rooftop PV entity clustering

This repository implements the probabilistic entity-embedding clustering method
from [Bölat et al. (2025)](https://arxiv.org/abs/2505.10699): daily PV profiles
are converted to words, embedded as Dirichlet distributions through LDA, then
clustered using statistical distances. Cluster quantiles support dataset
condensation and missing-value imputation.

`paper-v1` is the immutable tag for the historical paper implementation. The
current codebase is its UV-managed reproducibility and reuse successor. The
historical exploratory scripts and notebooks are intentionally available only
from that tag; the maintained interface is the package CLI and the executable
example notebook.

## Quick start

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --group dev
uv run entity-clustering run --config configs/studies/example.yaml --run-id example
uv run entity-clustering report --run-root runs/example/example
```

The bundled two-system dataset is a smoke-test example, so its configuration
uses one cluster. Outputs are written under `runs/<study>/<run-id>/` and are
ignored by Git.

## Paper workflow

The paper study needs the separately downloaded Utrecht source dataset. Place
`filtered_pv_power_measurements_ac.csv` and `metadata.csv` in `data/zenodo/`,
then create the derived daily input and run the declared sweep:

```bash
uv run entity-clustering prepare-data --config configs/bases/utrecht_zenodo_v1.yaml
uv run entity-clustering run --config configs/studies/paper_v1.yaml --run-id paper-v1
uv run entity-clustering report --run-root runs/paper_v1/paper-v1
```

The preparation command averages UTC one-minute AC values in 15-minute bins;
a bin with fewer than 15 observations is retained as missing. It records the
source checksums beside the derived CSV. The source dataset is not redistributed
here; cite and download it from its [Zenodo record](https://doi.org/10.5281/zenodo.6906504).

## Configuration and outputs

Scientific choices are split into four small YAML documents:

```text
configs/bases/        input data and source preparation
configs/methods/      LDA settings
configs/evaluations/  quantiles and score modes
configs/studies/      complete declared hyperparameter sweeps
```

Each run records its resolved configuration, SHA-256 configuration and input
hashes, Git commit/tag, installed package versions, seed, log, fitted models,
cluster assignments, scores, and skipped invalid trials. `--resume` is accepted
only when the stored resolved configuration hash is identical.

See [scientific documentation](docs/scientific/method_and_assumptions.md), the
[usage guide](docs/technical/usage_guide.md), and the
[configuration](docs/technical/configuration_reference.md) and
[artifact references](docs/technical/artifact_reference.md). Selected
historical paper figures are retained as
[reference artifacts](docs/reference/paper-v1/README.md).

## Validation

```bash
uv run ruff check src tests
uv run pytest
```

These checks use the small example and synthetic data; they do not download or
run the multi-gigabyte paper experiment.

## Citation and license

Please cite the method and source dataset as described in
[CITATION.cff](CITATION.cff). The source code is MIT-licensed; see
[LICENSE](LICENSE).
