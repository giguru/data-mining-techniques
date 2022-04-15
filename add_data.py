import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Dict
from tqdm import tqdm
from datetime import timedelta

def add_data_daily(data):
    df = data

    #setup variables
    last_break=0
    last_scr_td=timedelta(hours=0,minutes=0)
    sleeptime=0
    screen_counter=0

    #only look at screen data
    screendf=df[df['variable'].str.contains('screen')]
    #look at the first current date
    current_date=screendf.iloc[0]['time'].date()

    #iterate over screen data
    for screenstamp in screendf.iterrows():
        #increase screen watch counter
        screen_counter+=1
        
        #get the time
        screen_watch=(screenstamp[1]['time'])

        #if we're at a new date, ad a row to the dataframe to signify amount of screenwatches
        if current_date!=screen_watch.date():
            last_date=current_date
            d ={'id':[screenstamp[1]['id']],'time':[current_date],'variable':["amount_screen"],'value':[screen_counter]}
            dataf = pd.DataFrame(data=d)
            data=pd.concat([data,dataf], ignore_index=True)
            screen_counter=0
        #update current_date
        current_date=screen_watch.date()
        


    #iterate over screen data
    for screenstamp in screendf.iterrows():
        screen_watch=(screenstamp[1]['time'])


        current_date=screen_watch.date()
        scr_td = timedelta(hours=screen_watch.hour, minutes=screen_watch.minute)
        #after 9 pm, go to sleep
        if (screen_watch.hour)>=21:
            last_break=0
            sleeptime=1
        
        #if there's a last screen timedelta object, compare the two, if the difference between them is bigger than last_break, update it
        if last_scr_td:
            new_last_break=(scr_td-last_scr_td).seconds
            if new_last_break>last_break:
                last_break=new_last_break
        last_scr_td=timedelta(hours=screen_watch.hour, minutes=screen_watch.minute)
        
        #if it's waking time, we update the dataframe, and put sleeptime to 0.
        if screen_watch.hour>=10 and  screen_watch.hour<21 and sleeptime==1:
            sr=round(last_break/3600,2)
            d ={'id':[screenstamp[1]['id']],'time':[current_date],'variable':["screenrest"],'value':[sr]}
            dataf = pd.DataFrame(data=d)
            data=pd.concat([data,dataf], ignore_index=True)
            sleeptime=0
    return data


def read_data(**kwargs):
    dtypes = {}
    df = pd.read_csv('dataset_mood_smartphone.csv', dtype=dtypes, parse_dates=['time'],**kwargs)

    return df


data = read_data()


newdata=(add_data_daily(data))
newdata.to_csv('newdata.csv')