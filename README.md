# GeneralPort 
Improving Generalization of Language-Conditioned Robot Manipulation

[Project Page](https://qm-ipalab.github.io/GeneralPort/) 


## Installation
Setup virtualenv and install requirements:
```bash
# setup virtualenv with whichever package manager you prefer
virtualenv -p $(which python3.8) --system-site-packages generalport_env  
source generalport_env/bin/activate
pip install --upgrade pip

cd generalport
pip install -r requirements.txt

export CLIPORT_ROOT=$(pwd)
pip install -e .  # install cliport
```

Install SAM2:
```bash
git clone git@github.com:Cherry-first/sam2.git
cd sam2
pip install -e .
```

## Dataset Generation
Generate a `train` set of 1000 demonstrations for `stack-block-pyramid-seq-seen-colors` inside `$CLIPORT_ROOT/data`:
```bash
python cliport/demos.py n=1000 \
                        task=stack-block-pyramid-seq-seen-colors \
                        mode=train 
```

## Training, Validation and Testing
Training: submit train_clipfit.slurm or
```bash
python cliport/train0.py train.task=towers-of-hanoi-seq-seen-colors \
                        train.agent=clipfit\
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=20 \
                        train.n_steps=5000 \
                        train.save_steps=[120,200] \
                        train.exp_folder=exps_clipfit \
                        dataset.cache=False 
```

Validation: Submit val_clipfit.slum or
```bash
python cliport/eval0.py model_task=align-rope \
                       eval_task=align-rope \
                       agent=clipfit \
                       mode=val \
                       n_demos=25 \
                       train_demos=20 \
                       checkpoint_type=val_missing \
                       exp_folder=exps_clipfit \
```

Testing:
```bash
python cliport/eval0.py model_task=align-rope \
                       eval_task=align-rope \
                       agent=clipfit \
                       mode=test \
                       n_demos=25 \
                       train_demos=20 \
                       checkpoint_type=val_missing \
                       exp_folder=exps_clipfit \
```

