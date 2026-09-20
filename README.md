# Si3N4 Deep Learning Potential

## Project outline

The training framework of Si3N4 deep learning potential, two-stage strategy:

1. Stress-free training to obtain force/energy models
2. low stress fine-tuning, introducing stress prediction

## Data

* 5000 frames (4000 for training / 500 for validation / 501 for testing)
* Atomic number 252 (Si = 108, N = 144)
* Path: data/Si3N4_5000/

## Model architecture

* descriptor：se\_e2\_a
* Embedded network：\[25, 50, 100]
* Fitting network：\[200, 200, 200]
* cutoff radius：6.0 Å

## Two-phase training

### Phase One: Stress-Free Training

* configs：configs/no\_stress.json
* steps：50000
* Loss weight：pref\_e 0.02 to 1.0，pref\_f 1000 to 1.0，pref\_v = 0
* output：models/graph\_no\_stress\_new.pb

### Phase Two: Stress Fine-Tuning

* configs：configs/stress\_finetune.json
* steps：10000
* Loss weight：pref\_v 0.001 to 0.01
* input：graph\_no\_stress\_new.pb
* output：models/graph\_stress\_finetuned.pb

## Final accuracy (test set)

|indicator|Stress-free model|Fine-tuning model|
|-|-|-|
|Energy MAE/atom|0.287 meV|0.307 meV|
|Force MAE|0.00696 eV/A|0.00787 eV/A|
|Virial MAE/atom|Untrained|0.144 eV|

## Directory structure

data/
descriptor/
configs/
models/
checkpoints/
logs/
scripts/

## LAMMPS order

pair\_style deepmd models/graph\_stress\_finetuned.pb
pair\_coeff \* \* Si N

