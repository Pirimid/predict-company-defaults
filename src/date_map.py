import pandas as pd

DATE_MAP = {
    'FY-1': 2018,
    'FY-2': 2017,
    'FY-3': 2016,
    'FY-4': 2015,
    'FY-5': 2014,
    'FY-6': 2013,
    'FY-7': 2012,
    'FY-8': 2011,
    'FY-9': 2010,
    'FY-10': 2009,
    'FY-11': 2008,
    'FY-12': 2007,
    'FY-13': 2006,
    'FY-14': 2005,
}

def get_next_day(timeStamp):
    """
     If the current timeStamp is of weekends then returns the next working day.
     Arguments:
      * timeStamp: Pandas timeStamp
      * returns the next working day. 
    """
    if timeStamp.dayofweek == 5:
        return pd.to_datetime(f"{timeStamp.day + 2}/{timeStamp.month}/{timeStamp.year}", dayfirst=True)
    elif timeStamp.dayofweek == 6:
        return pd.to_datetime(f"{timeStamp.day + 1}/{timeStamp.month}/{timeStamp.year}", dayfirst=True)
    else:
        return timeStamp
        