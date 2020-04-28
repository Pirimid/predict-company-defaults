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

## Results/Report

We gathered data of more than 100 companies from different sectors. The data includes market data and annual reports published by companies. After going through many different companies’ data, we have gathered some evidence that there is always some pattern in the companies which are going default and which are non-default. For example, if a company is having an increasing debt over time with a little bit of increment in cash-flow or revenue then pretty much chances are that the company will go default. This is a really small example of having some kind of pattern in the data of the companies.

In our findings, we used many different techniques and tried to predict whether the company will go default or not. Following is the brief of our model training/testing process.

**Z-Score model** :- The classic z-score model was used for many decades to analyze companies’ default. The model uses some financial ratios to predict companies' health. We used this model and it was able to give 60% accuracy on our data. Here is the code for the Z-score model.

We used modern deep learning models, to begin with. We split our data into 80% training and 20% testing using stratification.

**LSTM**:-  We trained the LSTM model for 5 folds and we got the following results.

1. 0.73283374
2. 0.768442
3. 0.84433871
4. 0.84070730
5. 0.80934374

On average, the LSTM model gave 79.913% accuracy on testing data.

**Bidirectional LSTM**:- Bidirectional LSTM is the successor of LSTM models. It not only looks back into the data but looks into the future as well. This way in sequence to sequence data it can learn long term dependencies. In our experimental results, we noticed this clearly. In five-fold training, it gave following accuracy on the testing data,

1. 0.78634277
2. 0.7716389
3. 0.82728721
4. 0.8641510
5. 0.8555649

On average we saw that Bidirectional LSTM is 82% accurate in predicting the company default. It is very clear that Bidirectional LSTM performs better than LSTM.

We have done plotting of trainin and testing metrics, the images are in `img` directory.

Here is the first fold training metrics plots

![training_metric](/img/training_fold0.png)

Here is the accuracy plot for testing data for that fold.

![testing_acc](/img/test_fold0.png)
