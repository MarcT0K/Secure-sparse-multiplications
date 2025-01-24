# Codebase for "Secure sparse matrix multiplications and their applications to PPML"

The goal of this codebase is to implement the experiments mentionned in the paper. **It is not a secure sparse linear algebra package**.
Extra effort would be required to integrate these algorithms into MPC libraries such as MPyC.
Hence, our codebase only contains simplify classes to demonstrate algorithms and do not cover all possible operations (e.g., transpose) on sparse data structures.

## Installing the dependencies

First, you need to download the datasets used in some of our benchmark:

```
bash ./download_datasets.sh
```

Second, you should set up a Python virtual environment:

```
virtualenv .venv
source .venv/bin/activate
```

Finally, you can install our package:

```
pip3 install .
```

## Running the experiments

To run the "data-generation" experiments, you need to execute the following command (this should create a `data` folder):

```
run_all_experiments
```

Finally, you can generate the figures (this should create a `figure` folder):

```
generate_all_figures
```

