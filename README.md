# Sentiment analysis

This repository contains results of comparison of two models for sentiment analysis task.
The code can be run locally.

## Installation
Check that python 3.11 is installed. Install it if missing.

Install virtual environment
```
python3.11 -m venv venv
```
Activate it
```
. ./venv/bin/activate
```
Install requirements
```
pip install -r requirements.txt
```
Requirements installation can fail due to llama-cpp-python.
In this case check github readme https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file

Load models to the `models` directory

## Reproduce experiments
Run jupyter lab
```
jupyter lab
```
Open notebook `experiments.ipynb` and run all cells one by one.
