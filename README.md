# Black-Box Property Inference Attacks
## Setup
* Locally:
`pip install requirements.txt`
* Server:
```
conda create --name "pia-tf" python=3.10.9 ipython
conda activate pia-tf
conda install jupyter pip
pip install pandas numpy tensorflow keras-tuner
conda deactivate && python -m ipykernel install --user --name pia-tf
nano ~/.local/share/jupyter/kernels/pia-tf/kernel.json
```
(add the kernel manually in case it has not been added to list)

### Create synthetic data for genereting model output
* see `generate-synthetic-data.py`
* No need to re-run, result is included as `data/syn_data.csv`
### Create shadow model outputs to train adversary
* Training and using shadow models
* Run `generate-adv-input.py` -> might take a long time
### Create adversary
* Finetuning has been done in tune_adversary.ipynb
### Loading and using the adversary
* See `pia.ipynb`
