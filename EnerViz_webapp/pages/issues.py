import dash
from dash import html, dcc


from dash import Dash, html, dcc, Input, Output, State, callback
import pandas as pd
import dash
from dash import html
import plotly.graph_objs as go
from dash import html
from dash import dcc
import pandas as pd
import numpy as np
import dash
import dash.dependencies as dd
from io import BytesIO
from collections import Counter
from wordcloud import WordCloud
import base64
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import plotly.express as px
from dash import dash_table
from datetime import datetime
import time
import os
import sys
from matplotlib.colors import LinearSegmentedColormap

dash.register_page(__name__)

DATA_FOLDER = 'Datasets/classified'

df = pd.read_csv('Datasets/classified_data.csv')
df = pd.DataFrame(df)
#df = df.dropna()

df['preprocessed_comments'] = df['preprocessed_comments'].apply(str)
#######################PAGE 1#####################
#Count data
count_row = df.shape[0]
positive_count = (df['sentiment'] == 'positive').sum().sum()
negative_count = (df['sentiment'] == 'negative').sum().sum()


df['created_time'] = pd.to_datetime(df['created_time'])


# Define the available options for the first dropdown
options1 = [
    {'label': 'Billing', 'value': 'option1'},
    {'label': 'Power', 'value': 'option2'},
    {'label': 'Unforeseen Events', 'value': 'option3'}
]
# Define the available options for the second dropdown
options2 = {
    'option1': [
        {'label': 'Inaccurate', 'value': 'option1-1'},
        {'label': 'Delay', 'value': 'option1-2'},
        {'label': 'Increase Rates', 'value': 'option1-3'}
    ],
    'option2': [
        {'label': 'Power Interruptions', 'value': 'option2-1'},
        {'label': 'Intermittent Power', 'value': 'option2-2'},
        {'label': 'Line Tapping', 'value': 'option2-3'}
    ],
    'option3': [
        {'label': 'Natural Disasters', 'value': 'option3-1'},
        {'label': 'Unscheduled Repairs', 'value': 'option3-2'},
        {'label': 'Accidents', 'value': 'option3-3'}
    ]
}
# Define the keywords 
option1_allwords = []
option1_words = ['reading', 'maling reading', 'hindi tama', 'mali mali']
option1_words1 = ['walang bill', 'tagal bill', 'dumating bill', 'wala bill', 'wara bill', 'wra bill', 'wala bill']
option1_words2 = ['mahal bill', 'mahal singil', 'mahalon bayadan', 'mahalon singil', 'increase', 'kwh']

# Merge the keywords
option1_allwords.extend(option1_words)
option1_allwords.extend(option1_words1)
option1_allwords.extend(option1_words2)

option2_allwords = []
#option2_words = ['brownout', 'palsok', 'blackout','abiso', 'advisory', 'interruption', 'supply', 'power', 'wara', 'wra', 
#                 'wla', 'wala', 'mayo', 'mainitun', 'init', 'kuryente', 'ibalik', 'ilaw', 'negosyo', 'damay', 'lowbat', 
#                 'pakibalik', 'balik', 'extended', 'schedule', 'off']
option2_words = ['brownout', 'blackout', 'interruption',
                    'lowbat'
                    ]
option2_words1 = ['patay', 'sindi', 'puropalsok', 'palsok', 'palsuk', 'pawala', 'pakibalik', 'balik',
                  'palsok', 'paltok', 'purupaltok', 'wara power', 'wra power', 'wala power', 'wla power', 'mayo kuryente', 'wara ilaw', 'wala ilaw', 
                  'wra ilaw','wla ilaw', 'mainitun', 'init']
option2_words2 = ['jumper', 'illegal', 'illigal', 'iligal', 'illegal connection' ]

option2_allwords.extend(option2_words)
option2_allwords.extend(option2_words1)
option2_allwords.extend(option2_words2)

option3_allwords = []
option3_words = ['bagyo', 'uran']
option3_words1 = ['repair', 'ayuson', 'transformer', 'unscheduled', 'poste']
option3_words2 = ['sulo', 'sabog', 'putok']

option3_allwords.extend(option3_words)
option3_allwords.extend(option3_words1)
option3_allwords.extend(option3_words2)

defined_words = []

defined_words.extend(option1_allwords)
defined_words.extend(option2_allwords)
defined_words.extend(option3_allwords)

# Count the number of occurrences of each word in each comment and store the result in a new column called 'word_count'
df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_allwords]))
# Group the DataFrame by the 'created_time' and 'sentiment' columns and sum the count of each selected word for each date and sentiment
sentiment_filter = ['positive']
grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'raw_comments', 'sentiment'])['word_count'].sum().reset_index(name='count')
            # Filter the DataFrame to include only the rows that contain any of the selected words

# create a new column for each word in the dataset
for word in defined_words:
    df[word] = 0

# count the occurrences of each word in each sentence
for index, row in df.iterrows():
    sentence = row['preprocessed_comments']
    # Convert the sentence to a string if it's not already
    if isinstance(sentence, float):
        sentence = str(sentence)
    
    words = sentence.split()
    for word in words:
        if word in defined_words:
            df.at[index, word] += 1

# convert created_time column to datetime
df['created_time'] = pd.to_datetime(df['created_time'])

# create monthly count dataframe
df_monthly = df.groupby([pd.Grouper(key='created_time', freq='M')])[defined_words].sum()
df_monthly = df_monthly.reset_index()

# create daily count dataframe
df_daily = df.groupby([pd.Grouper(key='created_time', freq='D')])[defined_words].sum()
df_daily = df_daily.reset_index()

layout = html.Div(className='body', children=[
   
            html.Div(className='nav-main', children=[
        html.Div(children=[
            html.Img(className='nav-logo',src='/assets/logo.png'),
            html.Div(children=[
                html.Div(className='nav-title',children=['EnerViz']),
            ]),
        ], style={'display': 'flex'}),
        html.Div(className='nav-link-container',children=[
            html.A('Home', href='/', className='nav-link'),
            html.A('Classify', href='/classify', className='nav-link'),
            html.A('Dashboard', href='/dashboard', className='nav-link'),
            html.A('Issues', href='/issues', className='nav-link'),
            html.Img(className='nav-reload',src='/assets/refresh.png',id='update-button'),
           
        ]),
       
    ]),
    
        html.Div(id='', className='', children=[
                    html.Div(id='', className='', children=[
                     dcc.Checklist(
                    id='sentiment-checkbox',
                    options=[
                        {'label': 'Positive Sentiment', 'value': 'positive'},
                        {'label': 'Negative Sentiment', 'value': 'negative'},
                        {'label': 'All Sentiment', 'value': 'all'}
                    ],style={'width':'98%','margin':'20px'},
                    value=['all'],
                    labelStyle={'display': 'inline-block','margin':'10px'}
                ),
                   
                ],style={'width':'98%','margin':'20px'}),
    html.H4(children='Main Issues',style={'color': ''}),
    dcc.Dropdown(
        id='dropdown1',
        options=options1,
        value=options1[0]['value'] 
    ),
    html.H4(children='Sub-Issues'),
    dcc.Dropdown(
        id='dropdown2',
        options=options2[options1[0]['value']],
        value=options2[options1[0]['value']][0]['value']
    ),
    html.Br(id='', className='', children=[]),
    dcc.RangeSlider(
        id='date-slider',
        min=df_monthly['created_time'].min().timestamp(),
        max=df_monthly['created_time'].max().timestamp(),
        value=[df_monthly['created_time'].min().timestamp(), df_monthly['created_time'].max().timestamp()],
        marks={pd.Timestamp(month).timestamp(): month for month in df_monthly['created_time'].dt.strftime('%Y-%m').unique()},
        step=None
    ),

    dcc.Graph(
        id='word-count-graph'
    ),
    html.Div(id='total-counts'),
    html.Div(id='table-container'),
    ],style={'margin':'10px'}),
    
    ],style={
    'font-size': '1em',
    'font-weight': '400',
    'font-family': 'Arial',
    'color': 'rgb(50, 50, 50)',
    
    })




@callback(
    Output('dropdown2', 'options'),
    [Input('dropdown1', 'value')]
)
def update_dropdown2_options(selected_value):
    
    return options2[selected_value]
@callback(
    dash.dependencies.Output('word-count-graph', 'figure'),
    Output('table-container', 'children'),
    Output('total-counts', 'children'),
    dash.dependencies.Input('date-slider', 'value'),
    Input('dropdown2', 'value'),
    Input('dropdown1', 'value'),
    dash.dependencies.Input('sentiment-checkbox', 'value'),
    Input('update-button', 'n_clicks'),
)
def update_graph(date_range, value2, value1, sentiment,n_clicks):
    global df
    start_date = pd.to_datetime(date_range[0], unit='s')
    end_date = pd.to_datetime(date_range[1], unit='s')
    #########Graph for issues and sub-issues, filtered by sentiment
    data = []
    colors = {'positive': 'green', 'negative': 'red', 'all': '#23adfd'}
    if value1 == 'option1':

        if value2 == 'option1-1':
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option1_words 
            
        elif value2 == 'option1-2':
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words1]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']

                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option1_words1
            
        else:
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words2]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option1_words2
            
    elif value1 == 'option2':
        df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option2_allwords]))
        if 'positive' in sentiment:
            sentiment_filter = ['positive']
            grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
            grouped_df = grouped_df.fillna(0)
            grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
            grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
            data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
        if 'negative' in sentiment:
            sentiment_filter = ['negative']
            grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
            grouped_df = grouped_df.fillna(0)
            grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
            grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
            data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
        if 'all' in sentiment:
            sentiment_filter = ['positive', 'negative', 'neutral']
            grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
            grouped_df = grouped_df.fillna(0)
            grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
            grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
            data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
        selected_words = option2_allwords

        if value2 == 'option2-1':
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option2_words]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
            # Group the DataFrame by the 'created_time' and 'sentiment' columns, filtered by sentiment
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option2_words

        elif value2 == 'option2-2':
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option2_words1]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option2_words1

        else:
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option2_words2]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
            # Group the DataFrame by the 'created_time' and 'sentiment' columns, filtered by sentiment
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option2_words2

    elif value1 == 'option3':
        df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option3_allwords]))
        if 'positive' in sentiment:
            sentiment_filter = ['positive']
            grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
            grouped_df = grouped_df.fillna(0)
            grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
            grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
            data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
        if 'negative' in sentiment:
            sentiment_filter = ['negative']
            grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
            grouped_df = grouped_df.fillna(0)
            grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
            grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
            data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
        if 'all' in sentiment:
            sentiment_filter = ['positive', 'negative', 'neutral']
        # Group the DataFrame by the 'created_time' and 'sentiment' columns, filtered by sentiment
            grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
            grouped_df = grouped_df.fillna(0)
            grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
            grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
            data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
        selected_words = option3_allwords

        if value2 == 'option3-1':
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option3_words]))
            if sentiment == 'positive':
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if sentiment == 'negative':
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option3_words
            
        elif value2 == 'option3-2':
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option3_words1]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option3_words1
        else:
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option3_words2]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby('created_time')['word_count'].sum().reset_index(name='total_count')
                grouped_df = grouped_df.fillna(0)
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['total_count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option3_words2

    layout = go.Layout(title='Sentiment Analysis', xaxis=dict(title='Date'), yaxis=dict(title='Word Count'))
    layout.showlegend=False

    ######Table for comments 
    # Create a new column that contains the count of each selected word in each row
    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in selected_words]))

    # Filter the DataFrame to include only the rows that contain any of the selected words
    filtered_df = df[df['word_count'] > 0]

    # Filter the DataFrame to include only the comments within the selected date range
    filtered_df = filtered_df[(filtered_df['created_time'] >= start_date) & (filtered_df['created_time'] <= end_date)]
    
    # Filter the DataFrame to include only the comments with the selected sentiment
    if 'positive' in sentiment:
        sentiment_filter = ['positive']
    if 'negative' in sentiment:
        sentiment_filter = ['negative']
    if 'positive' in sentiment and 'negative' in sentiment:
        sentiment_filter = ['positive', 'negative']
    if 'all' in sentiment:
        sentiment_filter = ['positive', 'negative', 'neutral']
    filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]

    # Compute the total count of word counts in the comment list
    total_counts =  filtered_df['word_count'].sum()

    # Create a list of the comments that contain any of the selected words
    # Assuming that 'created_time' column is in datetime format
    filtered_df['created_time'] = filtered_df['created_time'].dt.strftime('%Y-%m-%d')
    comments_list = filtered_df[['created_time', 'raw_comments']][filtered_df['word_count'] > 0].values.tolist()
    comments_list = sorted(comments_list, key=lambda x: x[0])
        # Count the number of occurrences of each word in each comment and store the result in a new column called 'word_count'
    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_allwords]))
    # Group the DataFrame by the 'created_time' and 'sentiment' columns and sum the count of each selected word for each date and sentiment
    sentiment_filter = ['positive']
    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'raw_comments', 'sentiment'])['word_count'].sum().reset_index(name='count')
                # Filter the DataFrame to include only the rows that contain any of the selected words

    # create a new column for each word in the dataset
    for word in defined_words:
        df[word] = 0

    # count the occurrences of each word in each sentence
    for index, row in df.iterrows():
        sentence = row['preprocessed_comments']
        # Convert the sentence to a string if it's not already
        if isinstance(sentence, float):
            sentence = str(sentence)
        
        words = sentence.split()
        for word in words:
            if word in defined_words:
                df.at[index, word] += 1

    # convert created_time column to datetime
    df['created_time'] = pd.to_datetime(df['created_time'])

    # create monthly count dataframe
    df_monthly = df.groupby([pd.Grouper(key='created_time', freq='M')])[defined_words].sum()
    df_monthly = df_monthly.reset_index()

    # create daily count dataframe
    df_daily = df.groupby([pd.Grouper(key='created_time', freq='D')])[defined_words].sum()
    df_daily = df_daily.reset_index()



    # Create a data table to display the comments
    table = dash_table.DataTable(
        columns=[
            {'name': 'Created Time', 'id': 'created_time'},
            {'name': 'Comments', 'id': 'comment'}
        ],
        data=[{'created_time': comment[0], 'comment': comment[1]} for comment in comments_list],
        style_table={'height': '450px', 'overflowY': 'scroll'},
        style_cell={'textAlign': 'left', 'padding': '10px'}
    )

    # Create a div to display the total count of word counts in the comment list
    total_counts_div = html.Div(
        f"Total count: {total_counts}",
        style={'padding': '10px', 'fontWeight': 'bold', 'background':'white','border-radius':'5px', 'margin':'2px'}
    )






    if n_clicks is not None:
        
        df = pd.read_csv('Datasets/classified_data.csv')
        df = pd.DataFrame(df)
        #df = df.dropna()

        df['preprocessed_comments'] = df['preprocessed_comments'].apply(str)
        #######################PAGE 1#####################

        df['created_time'] = pd.to_datetime(df['created_time'])


        start_date = pd.to_datetime(date_range[0], unit='s')
        end_date = pd.to_datetime(date_range[1], unit='s')
        #########Graph for issues and sub-issues, filtered by sentiment
        data = []
        colors = {'positive': 'green', 'negative': 'red', 'all': '#23adfd'}
        if value1 == 'option1':

            if value2 == 'option1-1':
                if 'positive' in sentiment:
                    sentiment_filter = ['positive']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if 'negative' in sentiment:
                    sentiment_filter = ['negative']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                
            elif value2 == 'option1-2':
                if 'positive' in sentiment:
                    sentiment_filter = ['positive']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words1]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if 'negative' in sentiment:
                    sentiment_filter = ['negative']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words1]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words1]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                selected_words = option1_words1
                
            else:
                if 'positive' in sentiment:
                    sentiment_filter = ['positive']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words2]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if 'negative' in sentiment:
                    sentiment_filter = ['negative']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words2]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                    df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_words2]))
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                selected_words = option1_words2
                
        elif value1 == 'option2':
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option2_allwords]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                grouped_df = sorted(grouped_df, key=lambda x: x[0])
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                grouped_df = sorted(grouped_df, key=lambda x: x[0])
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                grouped_df = sorted(grouped_df, key=lambda x: x[0])
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option2_allwords

            if value2 == 'option2-1':
                df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option2_words]))
                if 'positive' in sentiment:
                    sentiment_filter = ['positive']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if 'negative' in sentiment:
                    sentiment_filter = ['negative']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                # Group the DataFrame by the 'created_time' and 'sentiment' columns, filtered by sentiment
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                selected_words = option2_words

            elif value2 == 'option2-2':
                df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option2_words1]))
                if 'positive' in sentiment:
                    sentiment_filter = ['positive']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if 'negative' in sentiment:
                    sentiment_filter = ['negative']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                selected_words = option2_words1

            else:
                df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option2_words2]))
                if 'positive' in sentiment:
                    sentiment_filter = ['positive']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if 'negative' in sentiment:
                    sentiment_filter = ['negative']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                # Group the DataFrame by the 'created_time' and 'sentiment' columns, filtered by sentiment
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                selected_words = option2_words2

        elif value1 == 'option3':
            df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option3_allwords]))
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                grouped_df = sorted(grouped_df, key=lambda x: x[0])
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                grouped_df = sorted(grouped_df, key=lambda x: x[0])
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
            if 'all' in sentiment:
                sentiment_filter = ['positive', 'negative', 'neutral']
            # Group the DataFrame by the 'created_time' and 'sentiment' columns, filtered by sentiment
                grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                grouped_df = sorted(grouped_df, key=lambda x: x[0])
                data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option3_allwords

            if value2 == 'option3-1':
                df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option3_words]))
                if sentiment == 'positive':
                    sentiment_filter = ['positive']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if sentiment == 'negative':
                    sentiment_filter = ['negative']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                selected_words = option3_words
                
            elif value2 == 'option3-2':
                df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option3_words1]))
                if 'positive' in sentiment:
                    sentiment_filter = ['positive']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if 'negative' in sentiment:
                    sentiment_filter = ['negative']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                selected_words = option3_words1
            else:
                df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option3_words2]))
                if 'positive' in sentiment:
                    sentiment_filter = ['positive']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                if 'negative' in sentiment:
                    sentiment_filter = ['negative']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                if 'all' in sentiment:
                    sentiment_filter = ['positive', 'negative', 'neutral']
                    grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    grouped_df['created_time'] = pd.to_datetime(grouped_df['created_time'])
                    grouped_df = grouped_df[(grouped_df['created_time'] >= start_date) & (grouped_df['created_time'] <= end_date)]
                    grouped_df = sorted(grouped_df, key=lambda x: x[0])
                    data.append(go.Scatter(x=grouped_df['created_time'], y=grouped_df['count'], fill='tozeroy', line=dict(color=colors['all'])))
                selected_words = option3_words2

        layout = go.Layout(title='Sentiment Analysis', xaxis=dict(title='Date'), yaxis=dict(title='Word Count'))
        layout.showlegend=False

        ######Table for comments 
        # Create a new column that contains the count of each selected word in each row
        df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in selected_words]))

        # Filter the DataFrame to include only the rows that contain any of the selected words
        filtered_df = df[df['word_count'] > 0]

        # Filter the DataFrame to include only the comments within the selected date range
        filtered_df = filtered_df[(filtered_df['created_time'] >= start_date) & (filtered_df['created_time'] <= end_date)]
        
        # Filter the DataFrame to include only the comments with the selected sentiment
        if 'positive' in sentiment:
            sentiment_filter = ['positive']
        if 'negative' in sentiment:
            sentiment_filter = ['negative']
        if 'positive' in sentiment and 'negative' in sentiment:
            sentiment_filter = ['positive', 'negative']
        if 'all' in sentiment:
            sentiment_filter = ['positive', 'negative', 'neutral']
        filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]

        # Compute the total count of word counts in the comment list
        total_counts =  filtered_df['word_count'].sum()

        # Create a list of the comments that contain any of the selected words
        # Assuming that 'created_time' column is in datetime format
        filtered_df['created_time'] = filtered_df['created_time'].dt.strftime('%Y-%m-%d')
        comments_list = filtered_df[['created_time', 'raw_comments']][filtered_df['word_count'] > 0].values.tolist()
        comments_list = sorted(comments_list, key=lambda x: x[0])



        # Count the number of occurrences of each word in each comment and store the result in a new column called 'word_count'
        df['word_count'] = df['preprocessed_comments'].str.lower().apply(lambda x: sum([x.count(word) for word in option1_allwords]))
        # Group the DataFrame by the 'created_time' and 'sentiment' columns and sum the count of each selected word for each date and sentiment
        sentiment_filter = ['positive']
        grouped_df = df[df['sentiment'].isin(sentiment_filter)].groupby(['created_time', 'raw_comments', 'sentiment'])['word_count'].sum().reset_index(name='count')
                    # Filter the DataFrame to include only the rows that contain any of the selected words

        # create a new column for each word in the dataset
        for word in defined_words:
            df[word] = 0

        # count the occurrences of each word in each sentence
        for index, row in df.iterrows():
            sentence = row['preprocessed_comments']
            # Convert the sentence to a string if it's not already
            if isinstance(sentence, float):
                sentence = str(sentence)
            
            words = sentence.split()
            for word in words:
                if word in defined_words:
                    df.at[index, word] += 1

        # convert created_time column to datetime
        df['created_time'] = pd.to_datetime(df['created_time'])

        # create monthly count dataframe
        df_monthly = df.groupby([pd.Grouper(key='created_time', freq='M')])[defined_words].sum()
        df_monthly = df_monthly.reset_index()

        # create daily count dataframe
        df_daily = df.groupby([pd.Grouper(key='created_time', freq='D')])[defined_words].sum()
        df_daily = df_daily.reset_index()



        # Create a data table to display the comments
        table = dash_table.DataTable(
            columns=[
                {'name': 'Created Time', 'id': 'created_time'},
                {'name': 'Comments', 'id': 'comment'}
            ],
            data=[{'created_time': comment[0], 'comment': comment[1]} for comment in comments_list],
            style_table={'height': '450px', 'overflowY': 'scroll'},
            style_cell={'textAlign': 'left', 'padding': '10px'}
        )

        # Create a div to display the total count of word counts in the comment list
        total_counts_div = html.Div(
            f"Total count: {total_counts}",
            style={'padding': '10px', 'fontWeight': 'bold', 'background':'white','border-radius':'5px', 'margin':'2px'}
        )
    return {'data': data, 'layout': layout}, table, total_counts_div