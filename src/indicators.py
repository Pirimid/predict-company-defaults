import talib
import pandas as pd

from src.logger import LOGGER

def enrich_data(data):
    """
    Enhances data frame with information on indicators and price patterns. Indicators and patterns are align so
    that they represent details for previous minute.

    :param data: DataFrame
    :return: DataFrame
    """
    LOGGER.info("Adding TA-Lib indicators as features...")
    # We specifically do shifting here so that all additional data represents information about past history.
    df = pd.concat((data, get_indicators(data).shift(), get_price_patterns(data).shift()), axis=1)
    df = df.fillna(method='ffill')
    return df.dropna()


def get_indicators(data, intervals=(5, 10, 20, 50, 100)):
    """
    Computes technical indicators given ticks data.
    These indicators are computed with fixed parameters, i.e. intervals argument shouldn't affect them:
    * Parabolic SAR
    * Chaikin A/D Line
    * On Balance Volume
    * Hilbert Transform - Instantaneous Trendline
    * Hilbert Transform - Trend vs Cycle Mode
    * Hilbert Transform - Dominant Cycle Period
    * Hilbert Transform - Dominant Cycle Phase
    * Typical Price
    These indicators are computed for each of periods given in intervals argument:
    * Exponential Moving Average
    * Double Exponential Moving Average
    * Kaufman Adaptive Moving Average
    * Midpoint Price over period
    * Triple Exponential Moving Average
    * Average Directional Movement Index
    * Aroon
    * Commodity Channel Index
    * Momentum
    * Rate of change Percentage: (price-prevPrice)/prevPrice
    * Relative Strength Index
    * Ultimate Oscillator (based on T, 2T, 3T periods)
    * Williams' %R
    * Normalized Average True Range
    * Time Series Forecast (linear regression)
    * Bollinger Bands
    For more details see TA-lib documentation.
    When there are options in indicator API, Close Price prices are used for computation. For volume TickVol is used.
    Note that some of the indicators are not stable and could output unexpected results if fed with NaNs or long series.

    :param trader DataFrame with data.
    :param intervals Iterable with time periods to use for computation.
                     Periods should be in the same sample units as ticks data, i.e. in minutes.
                     Default values: 5, 10, 20, 50 and 100 minutes.
    :return DataFrame with indicators. For interval-based indicators, interval is mentioned in column name, e.g. CCI_5.
    """
    indicators = {}
    # Time period based indicators.
    for i in intervals:
        indicators['DEMA_{}'.format(i)] = talib.DEMA(
            data['Close Price'], timeperiod=i)
        indicators['EMA_{}'.format(i)] = talib.EMA(data['Close Price'], timeperiod=i)
        indicators['KAMA_{}'.format(i)] = talib.KAMA(
            data['Close Price'], timeperiod=i)
        indicators['MIDPRICE_{}'.format(i)] = talib.MIDPRICE(
            data['High Price'], data['Low Price'], timeperiod=i)
        indicators['T3_{}'.format(i)] = talib.T3(data['Close Price'], timeperiod=i)
        indicators['ADX_{}'.format(i)] = talib.ADX(
            data['High Price'], data['Low Price'], data['Close Price'], timeperiod=i)
        indicators['AROON_down_{}'.format(i)], indicators['AROON_up_{}'.format(i)] = talib.AROON(
            data['High Price'], data['Low Price'], timeperiod=i)
        indicators['CCI_{}'.format(i)] = talib.CCI(
            data['High Price'], data['Low Price'], data['Close Price'], timeperiod=i)
        indicators['MOM_{}'.format(i)] = talib.MOM(data['Close Price'], timeperiod=i)
        indicators['ROCP_{}'.format(i)] = talib.ROCP(
            data['Close Price'], timeperiod=i)
        indicators['RSI_{}'.format(i)] = talib.RSI(data['Close Price'], timeperiod=i)
        indicators['ULTOSC_{}'.format(i)] = talib.ULTOSC(data['High Price'], data['Low Price'], data['Close Price'],
                                                         timeperiod1=i, timeperiod2=2 * i, timeperiod3=4 * i)
        indicators['WILLR_{}'.format(i)] = talib.WILLR(
            data['High Price'], data['Low Price'], data['Close Price'], timeperiod=i)
        indicators['NATR_{}'.format(i)] = talib.NATR(
            data['High Price'], data['Low Price'], data['Close Price'], timeperiod=i)
        indicators['TSF_{}'.format(i)] = talib.TSF(data['Close Price'], timeperiod=i)
        indicators['BBANDS_upper_{}'.format(i)], indicators['BBANDS_middle_{}'.format(i)], indicators['BBANDS_Low Priceer_{}'.format(i)] = talib.BBANDS(
            data['Close Price'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        indicators['ATR_{}'.format(i)] = talib.ATR(
            data['High Price'], data['Low Price'], data['Close Price'], timeperiod=i)
        indicators['NATR_{}'.format(i)] = talib.NATR(
            data['High Price'], data['Low Price'], data['Close Price'], timeperiod=i)
        indicators['BETA_{}'.format(i)] = talib.BETA(
            data['High Price'], data['Low Price'], timeperiod=i)
        indicators['CORREL_{}'.format(i)] = talib.CORREL(
            data['High Price'], data['Low Price'], timeperiod=1)
        indicators['LINEARREG_ANGLE_{}'.format(i)] = talib.LINEARREG_ANGLE(
            data['Close Price'], timeperiod=i)
        indicators['Close Price_STDDEV_{}'.format(i)] = talib.STDDEV(
            data['Close Price'], timeperiod=i)
        indicators['High Price_STDDEV_{}'.format(i)] = talib.STDDEV(
            data['High Price'], timeperiod=i)
        indicators['Low Price_STDDEV_{}'.format(i)] = talib.STDDEV(
            data['Low Price'], timeperiod=i)
        indicators['Open Price_STDDEV_{}'.format(i)] = talib.STDDEV(
            data['Open Price'], timeperiod=i)
        indicators['Close Price_VAR_{}'.format(i)] = talib.VAR(
            data['Close Price'], timeperiod=i, nbdev=1)
        indicators['Open Price_VAR_{}'.format(i)] = talib.VAR(
            data['Open Price'], timeperiod=i, nbdev=1)
        indicators['High Price_VAR_{}'.format(i)] = talib.VAR(
            data['High Price'], timeperiod=i, nbdev=1)
        indicators['Low Price_VAR_{}'.format(i)] = talib.VAR(
            data['Low Price'], timeperiod=i, nbdev=1)
    # Indicators that do not depend on time periods.
    indicators['Close Price_macd'], indicators['Close Price_macdsignal'], indicators['Close Price_macdhist'] = talib.MACD(
        data['Close Price'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['Open Price_macd'], indicators['Open Price_macdsignal'], indicators['Open Price_macdhist'] = talib.MACD(
        data['Open Price'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['High Price_macd'], indicators['High Price_macdsignal'], indicators['High Price_macdhist'] = talib.MACD(
        data['High Price'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['Low Price_macd'], indicators['Low Price_macdsignal'], indicators['Low Price_macdhist'] = talib.MACD(
        data['Low Price'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['SAR'] = talib.SAR(data['High Price'], data['Low Price'])
    indicators['HT_TRENDLINE'] = talib.HT_TRENDLINE(data['Close Price'])
    indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(data['Close Price'])
    indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(data['Close Price'])
    indicators['HT_DCPHASE'] = talib.HT_DCPHASE(data['Close Price'])
    indicators['TYPPRICE'] = talib.TYPPRICE(
        data['High Price'], data['Low Price'], data['Close Price'])
    return pd.DataFrame(indicators)


def get_price_patterns(data):
    """
    Detects common price patterns using TA-lib, e.g. Two Crows, Belt-hold, Hanging Man etc.

    :param data: DataFrame with data.
    :return: DataFrame with pattern "likelihoods" on -200 - 200 scale.
    """
    patterns = {name: getattr(talib, name)(data['Open Price'], data['High Price'], data['Low Price'], data['Close Price'])
                for name in talib.get_function_groups()['Pattern Recognition']}
    return pd.DataFrame(patterns)
