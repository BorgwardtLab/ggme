# ggme: <u>G</u>raph <u>G</u>enerative <u>M</u>odel <u>E</u>valuation
This is the official repository for the ICLR 2022 paper "Evaluation Metrics for Graph Generative Models: Problems, Pitfalls, and Practical Solutions" https://openreview.net/forum?id=tBtoZYKd9n

# Dependencies

Dependencies are managed using `poetry.` To setup the environment,
please run `poetry install` from the main directory (assuming the user
already has installed `poetry`).

# Running ggme

The primary script is contained in `main.py`. We assume that the user
has two distributions which they would like to compare using MMD, given
a specified kernel and descriptor function. 

We assume that each distribution of graphs is stored as a list of `networkx`
graphs. 

# Example script  

We provide an example run in `main.py` based on predictions of a graph
generative model and the graphs in the corresponding test set. To run
this, execute the following code from the main directory.

```bash
cd src
poetry run python main.py
```

# Citing our work

Please consider citing our work: 

```bibtex
@inproceedings{
	o'bray2022evaluation,
	title={Evaluation Metrics for Graph Generative Models: Problems, Pitfalls, and Practical Solutions},
	author={Leslie O'Bray and Max Horn and Bastian Rieck and Karsten Borgwardt},
	booktitle={International Conference on Learning Representations},
	year={2022},
	url={https://openreview.net/forum?id=tBtoZYKd9n}
}
```
