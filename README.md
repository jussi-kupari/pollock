First, download the Pollock repo dev branch from here

```bash
git clone --branch dev https://github.com/jussi-kupari/pollock
```

Then, create a micromamba environment and install pollock using the commands below.

```bash
cd pollock
micromamba create --prefix ./env
micromamba activate ./env
micromamba install -c pytorch -c conda-forge -y pytorch torchvision torchaudio cpuonly captum scanpy umap-learn ipykernel
pip install .
```
