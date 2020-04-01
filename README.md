# predict-company-defaults
Analyzing company's financials, stock price history, news, executive profiles and predict health of the company

## set up
1. Install `python 3.7`.
2. Use `pip install -r requirments.txt` to install dependecies. 

## Usage
1. Change the configuration in the `main.py` as per the need.
    1. Models supported are `LSTM`, `Dense`, `BiDirectional LSTM`.
    2. Modes supported are `train`, `test`, and `train_test`.
    3. Optimizer of your choice.
    4. Name of the data files to load.

    Above are the crucial one. All other arguments which can be passed to `trainer` can be changed from configuration. 
2. Run `python main.py`

## TODO
1. Results/Report
