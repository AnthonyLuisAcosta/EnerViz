import dash
from dash import html, dcc

dash.register_page(__name__)

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


DATA_FOLDER = 'Datasets/classified'

# Get the list of files in the data folder

# Define the container styles
container_style = {
    'display': 'inline-block',
    'width': 'calc(25% - 20px)',
    'height': '100px',  # Set the height of each container to 200px
    'margin': '10px',  # Add a 10px margin around each container
    'border-radius': '15px',
    'font-size': '70%',
    
}

df = pd.read_csv('Datasets/classified_data.csv')
df = pd.DataFrame(df)
#df = df.dropna()

df['preprocessed_comments'] = df['preprocessed_comments'].apply(str)
#######################PAGE 1#####################
#Count data
count_row = df.shape[0]
positive_count = (df['sentiment'] == 'positive').sum().sum()
negative_count = (df['sentiment'] == 'negative').sum().sum()

# convert created_time column to datetime
df['created_time'] = pd.to_datetime(df['created_time'])

# count the number of values in each date
ds = df['created_time'].value_counts().reset_index()
ds.columns = ['date', 'count']
sorted(ds['date'])
dates = pd.date_range(start=ds['date'].min() ,end=ds['date'].max(), freq='D')
ds = pd.DataFrame({'date': ds['date'], 'count': ds['count']})



def unixTimeMillis(dt):
    ''' Convert datetime to unix timestamp '''
    return int(dt.timestamp())

def unixToDatetime(unix):
    ''' Convert unix timestamp to datetime. '''
    return pd.to_datetime(unix, unit='s')

def getMarks(dates, Nth=100):
    ''' Returns the marks for labeling. 
        Every Nth value will be used.
    '''
    result = {}
    years = pd.DatetimeIndex(dates).year.unique()  # Get unique years from dates
    for i, year in enumerate(years):
        date = pd.to_datetime(f'{year}')  # Use January 1st of each year as mark
        result[unixTimeMillis(date)] = str(date.strftime('%Y-%m-%d'))
    return result

senti = df['sentiment'].value_counts().reset_index()
senti.columns = ['sentiment', 'count']
df['created_time'] = pd.to_datetime(df['created_time'])

###Sentiment Timeline
#Converting 'positive' as positive and 'negative' as negative 
#df['sentiment'] = df['sentiment'].apply(lambda x: 'positive' if x == 'positive' else 'negative')
positive_df = df[df['sentiment'] == 'positive']
negative_df = df[df['sentiment'] == 'negative']

positive_grouped = positive_df.groupby([ds['date']]).size().reset_index(name='count')
negative_grouped = negative_df.groupby([ds['date']]).size().reset_index(name='count')

# count the number of values in each date
counts = df['created_time'].value_counts().reset_index()
counts.columns = ['date', 'count']

#Get the time frame of data
dates = sorted(counts['date'])
start = str(dates[0])[:4]
end = str(dates[len(dates)-1])[:4]
duration = start + ' - ' + end

#Timeline Graph
df['date'] = pd.to_datetime(df['created_time'])

# Define colors for each sentiment
colors = {'positive': 'green', 'negative': 'red', 'all': '#23adfd'}
#df = df.drop(df.columns[1], axis=1)

df = df.drop('Unnamed: 0.1', axis=1, errors='ignore')


# convert created_time column to datetime
df['created_time'] = pd.to_datetime(df['created_time'])


label_value = 'all'
global BB

@callback(
    Output('file-dropdown', 'options'),
    Input('update-button', 'n_clicks'),
    State('file-dropdown', 'value')
)
def update_dropdown_options(n_clicks, selected_file):
    file_list = os.listdir(DATA_FOLDER)
    options = [{'label': file, 'value': file} for file in file_list]
    return options

####Dashboard layout
layout= html.Div(className='body', children=[
    
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
    html.Br(id='', className='', children=[]),
    html.Label(id='', className='', children='Select CSV file: ',style={'margin':'15px'}),
      dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': f, 'value': f} for f in os.listdir(DATA_FOLDER)],
        placeholder='Select a CSV file and visualize data', style = {'margin':'5px'}
    ),
    html.Div(children = [
    html.Div(id='update-output'),
    html.Div(
        children=[
            
             html.Img(src= '/assets/total.png',
                         style={'float': 'right', 'height': '100px','padding-right': '20%', 'opacity':'50%'}),
            html.Div(children=[
                html.H5(children='Total',
                        style={'text-align': 'center', 'color': 'white','text-align': 'left','margin': '20px 5px 0px 40px','opacity':'70%'}),
                html.H1(id = 'total-sentiment',
                        style={'text-align': 'left', 'margin': '0px 40px', 'font-weight': '900'}),
            ])
        ],
        style={**container_style, 'background': '#23adfd',
               'box-shadow': '2px 5px 8px rgba(0, 0, 0, 0.1), 2px 2px 5px rgba(0,0,0,0.1)', 'display': 'inline-block','overflow': 'hidden' }
    ),
    
    html.Div(
        children=[
            
             html.Img(src= '/assets/pos.png',
                         style={'float': 'right', 'height': '70px','padding-right': '20%','padding-top': '20px', 'opacity':'50%'}),
            html.Div(children=[
                html.H5(children='Positive Sentiments',
                        style={'text-align': 'center', 'color': 'white','text-align': 'left','margin': '20px 5px 0px 40px','opacity':'70%'}),
                html.H1(id = 'positive-sentiment',
                        style={'text-align': 'left', 'margin': '0px 40px', 'font-weight': '900'}),
            ])
        ],
        style={**container_style, 'background': '#23adfd',
               'box-shadow': '2px 5px 8px rgba(0, 0, 0, 0.1), 2px 2px 5px rgba(0,0,0,0.1)', 'display': 'inline-block','overflow': 'hidden' }
    ),
     html.Div(
        children=[
            
            html.Img(src= '/assets/neg.png',
                         style={'float': 'right', 'height': '70px','padding-right': '20%','padding-top': '20px', 'opacity':'50%'}),
            html.Div(children=[
                html.H5(children='Negative Sentiments',
                        style={'text-align': 'center', 'color': 'white','text-align': 'left','margin': '20px 5px 0px 40px','opacity':'70%'}),
                html.H1(id = 'negative-sentiment',
                        style={'text-align': 'left', 'margin': '0px 40px', 'font-weight': '900',}),
            ])
        ],
        style={**container_style, 'background': '#23adfd',
               'box-shadow': '2px 5px 8px rgba(0, 0, 0, 0.1), 2px 2px 5px rgba(0,0,0,0.1)', 'display': 'inline-block','overflow': 'hidden' }
    ),
        html.Div(
        children=[
            
             html.Img(src= '/assets/duration.png',
                         style={'float': 'right', 'height': '60px','padding-right': '20%','padding-top': '20px', 'opacity':'50%'}),
            html.Div(children=[
                html.H5(children='Duration',
                        style={'text-align': 'center','color': 'white','text-align': 'left','margin': '20px 5px 0px 40px','opacity':'70%'}),
                html.H1(id = 'year_range',
                        style={'text-align': 'left', 'margin': '0px 40px', 'font-weight': '900', 'font-size': '180%',}),
            ])
        ],
        style={**container_style, 'background': '#23adfd',
               'box-shadow': '2px 5px 8px rgba(0, 0, 0, 0.1), 2px 2px 5px rgba(0,0,0,0.1)', 'display': 'inline-block','overflow': 'hidden' }
    ),
    html.Div(
        children=[
            
            html.Div(children=[
                 dcc.Graph(id = 'word-freq-graph', style={'margin': '-4px 0px 0px -20px','overflow': 'hidden'}),
                 ], style={'height': '300px','overflow': 'hidden',
                    }),
           
    
        ],
        style={'background-color': '#23adfd', 'height': '300px', 'line-height': '5x', 'text-align': 'center',
               'box-shadow': '2px 5px 8px rgba(0, 0, 0, 0.1), 2px 2px 5px rgba(0,0,0,0.1)', 'border-radius': '15px',
               'width': 'calc(20% - 20px)','display': 'inline-block','float':'left','margin':'10px','overflow': 'hidden' }
    ),
    html.Div(
        children=[
         
            dcc.Dropdown(
                    id='label-dropdown',
                    options=[{'label': 'All', 'value': 'all'},{'label': 'Positive', 'value': 'positive'},{'label': 'Negative', 'value': 'negative'}],
                    value=label_value,style={'width':'180px', 'margin': '10px','font-size': '15px'},
                ),
    html.Img(id='wordcloud', src='')
        ],
        style={'background-color': '#ffffff', 'height': '300px', 'line-height': '5x', 'text-align': 'center',
               'box-shadow': '2px 5px 8px rgba(0, 0, 0, 0.1), 2px 2px 5px rgba(0,0,0,0.1)', 'border-radius': '15px',
               'width': 'calc(55% - 20px)','display': 'inline-block','float':'left','margin':'10px','overflow': 'hidden'}
    ),
    html.Div(id='', className='Sentiment Pie Chart', children=[
        html.Div(id='', className='', children='Sentiment Pie Chart',style = {'position':'relative', 'z-index': '2400','margin': '20px 0px 0px 0px'}),
        dcc.Graph(
            id='pie-chart',
            style={
                'width': '100%',
                'height': '140%',
                'margin': '-85px 0 0 0',
                'padding': '0',
            }
        ),
        
    ],
             style={'background-color': '#ffffff', 'height': '300px', 'line-height': '5px', 'text-align': 'center',
               'box-shadow': '2px 5px 8px rgba(0, 0, 0, 0.1), 2px 2px 5px rgba(0,0,0,0.1)', 'border-radius': '15px',
               'width': 'calc(25% - 20px)','display': 'inline-block','float':'right', 'margin':'10px','overflow': 'hidden'},),
        html.Div(
        children=[
            
            html.Div([
                html.H3('Sentiment Analysis Timeline'),
               
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
                 dcc.Graph(id='sentiment-graph'),
                html.Div([
                dcc.RangeSlider(
                id='date-slider',
                min=df['date'].min().timestamp(),
                max=df['date'].max().timestamp(),
                value=[df['date'].min().timestamp(), df['date'].max().timestamp()],
                marks={int(pd.Timestamp(date).timestamp()): {'label': pd.Timestamp(date).strftime('%b %Y'), 'style': {'transform': 'rotate(45deg)'}} for date in pd.date_range(start=df['date'].min().date(), end=df['date'].max().date(), freq='M').unique()},
                step=None
            )
                ], style={'margin': '50px'})
            ],style={'width':'98%','margin':'20px'})
                        
            
        ],
        style={'background-color': '#ffffff', 'height': '50%', 'line-height': '5x', 'text-align': 'center',
               'box-shadow': '2px 5px 8px rgba(0, 0, 0, 0.1), 2px 2px 5px rgba(0,0,0,0.1)', 'border-radius': '15px',
               'width': '99%','display': 'inline-block','margin':'10px'}
    ),
        
], style={'font-size': '1.5em',}),
], style={
    
    'font-weight': '400',
    'font-family': 'Arial',
    'color': 'rgb(50, 50, 50)',
    'background-color': '',
    'zoom': '100%'})



# Page 1 callbacks
# Define the callbacks
@callback(
    [Output('total-sentiment', 'children'),
     Output('positive-sentiment', 'children'),
     Output('negative-sentiment', 'children'),
     Output('year_range', 'children'),
     Output('pie-chart', 'figure'),
     Output('word-freq-graph', 'figure'),
     dd.Output('wordcloud', 'src'),
     dash.dependencies.Output('sentiment-graph', 'figure')],
    [Input('update-button', 'n_clicks'),
     dd.Input('label-dropdown', 'value'),
     dash.dependencies.Input('sentiment-checkbox', 'value'),
     dash.dependencies.Input('date-slider', 'value'),Input('file-dropdown', 'value')])


def update_sentiment(n_clicks, label_value, sentiment, timestamp_range, selected_file):
    from matplotlib.colors import LinearSegmentedColormap
    clrs = ['#23adfd', '#0068B0','#323232']  # Blue to black
    cmapname = 'blue_to_black'
    BB = LinearSegmentedColormap.from_list(cmapname, clrs)
    
    global df
    total_sentiment = len(df)
    total_sentiment = "{:,}".format(total_sentiment)
    #Total sentiments from the dataframe
    positive_sentiment = len(df[df['sentiment'] == 'positive'])
    positive_sentiment = "{:,}".format(positive_sentiment)
    negative_sentiment = len(df[df['sentiment'] == 'negative'])
    negative_sentiment= "{:,}".format(negative_sentiment)
    
    
    #Define min and max date from the dataframe
    mindate = str(df['created_time'].min())[0:4]
    maxdate = str(df['created_time'].max())[0:4]
    year_range = str(mindate + ' - ' + maxdate)
    
    
    piecolors = ['#23adfd', '#0068B0','#002B69']
    
    sentiment_count = df['sentiment'].value_counts()
    sentiment_count = sentiment_count.rename(index={'negative': 'negative', 'positive': 'positive'})
    sentiment_count = sentiment_count.rename_axis('sentiment').reset_index(name='count')

    labels = sentiment_count['sentiment']
    values = sentiment_count['count']

    #Word Frequency Bar Graph
    # Use pandas' value_counts function to get the word frequencies
    from nltk.corpus import stopwords

    # Define a list of stopwords
    stop_words = set(stopwords.words('english'))

    # Add custom stopwords to the list
    custom_stop_words = ['na', 'may','digdi','talaga']
    stop_words.update(custom_stop_words)
    # Create pie chart
    Piefig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker=dict(colors=piecolors))])
    
    
    # Use pandas' value_counts function to get the word frequencies, excluding the stop words
    word_freq = df["preprocessed_comments"].apply(lambda x: [word for word in x.split() if word not in stop_words]).explode().value_counts()[:10]

    # Convert the resulting Series back to a DataFrame and sort in descending order
    word_freq_df = word_freq.to_frame().reset_index()
    word_freq_df.columns = ["word", "frequency"]
    word_freq_df = word_freq_df.sort_values(by="frequency", ascending=True)

    #Word Cloud
    # Define a function to get the top words for a given label value
    DEFAULT_STOPWORDS = set(['may','na','talaga','digdi'])

    def get_top_words(label_value, stopwords=DEFAULT_STOPWORDS):
        if label_value == 'all':
            words_list = df['preprocessed_comments'].astype(str)
        else:
            words_list = df.loc[df['sentiment'] == label_value, 'preprocessed_comments'].astype(str)
        cnt = Counter()
        for text in words_list.values:
            for word in text.split():
                if word not in stopwords:
                    cnt[word] += 1  
        return dict(cnt.most_common(50))

    # Define the initial label value
    # Define a blue-to-black color map
    
    top_words = get_top_words(label_value)
    wc = WordCloud(background_color='white', width=900, height=230, colormap=BB )
    wc.generate_from_frequencies(frequencies=top_words)
    wc_image = wc.to_image()
    img = BytesIO()
    wc_image.save(img, format='PNG')
    encoded_image = base64.b64encode(img.getvalue()).decode()
    wordcloudimg = 'data:image/png;base64,{}'.format(encoded_image)
    
        
        
    df['created_time'] = pd.to_datetime(df['created_time'])

    # count the number of values in each date
    ds = df['created_time'].value_counts().reset_index()
    ds.columns = ['date', 'count']
    sorted(ds['date'])
    dates = pd.date_range(start=ds['date'].min() ,end=ds['date'].max(), freq='D')
    ds = pd.DataFrame({'date': ds['date'], 'count': ds['count']})

    #Timeline Graph
    df['date'] = pd.to_datetime(df['created_time'])
    
    # Set the range slider bounds and marks based on the date range in the CSV file
    min_date = df["date"].min().timestamp()
    max_date = df["date"].max().timestamp()
    marks = {int(pd.Timestamp(date).timestamp()): {"label": pd.Timestamp(date).strftime("%b %Y"), "style": {"transform": "rotate(45deg)"}} for date in pd.date_range(start=df["date"].min().date(), end=df["date"].max().date(), freq="M").unique()}
    
    # Set the initial range slider value to the full date range of the CSV file
    value = [min_date, max_date]

    # Define colors for each sentiment
    colors = {'positive': 'green', 'negative': 'red', 'all': '#23adfd'}

    def unixTimeMillis(dt):
        ''' Convert datetime to unix timestamp '''
        return int(dt.timestamp())

    def unixToDatetime(unix):
        ''' Convert unix timestamp to datetime. '''
        return pd.to_datetime(unix, unit='s')

    def getMarks(dates, Nth=100):
        ''' Returns the marks for labeling. 
            Every Nth value will be used.
        '''
        result = {}
        years = pd.DatetimeIndex(dates).year.unique()  # Get unique years from dates
        for i, year in enumerate(years):
            date = pd.to_datetime(f'{year}')  # Use January 1st of each year as mark
            result[unixTimeMillis(date)] = str(date.strftime('%Y-%m-%d'))
        return result

    senti = df['sentiment'].value_counts().reset_index()
    senti.columns = ['sentiment', 'count']
    df['created_time'] = pd.to_datetime(df['created_time'])

    ###Sentiment Timeline
    #Converting 'positive' as positive and 'negative' as negative 
    #df['sentiment'] = df['sentiment'].apply(lambda x: 'positive' if x == 'positive' else 'negative')
    positive_df = df[df['sentiment'] == 'positive']
    negative_df = df[df['sentiment'] == 'negative']

    positive_grouped = positive_df.groupby([ds['date']]).size().reset_index(name='count')
    negative_grouped = negative_df.groupby([ds['date']]).size().reset_index(name='count')

    # count the number of values in each date
    counts = df['created_time'].value_counts().reset_index()
    counts.columns = ['date', 'count']

    #Get the time frame of data
    dates = sorted(counts['date'])
    start = str(dates[0])[:4]
    end = str(dates[len(dates)-1])[:4]
    duration = start + ' - ' + end


    start_date = pd.to_datetime(timestamp_range[0], unit='s')
    end_date = pd.to_datetime(timestamp_range[1], unit='s')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    title = f'Timeline of Sentiment Counts from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
    
    ###Filtering count based from the selected sentiment e.g 'positive'
    if 'positive' in sentiment:
        filtered_positive = df[df['sentiment'] == 'positive']
        filtered_positive = filtered_positive.groupby(['date']).size().reset_index(name='count')
        filtered_positive = filtered_positive[(filtered_positive['date'] >= start_date) & (filtered_positive['date'] <= end_date)]
        filtered_positive = pd.merge(date_range.to_frame(name='date'), filtered_positive, on='date', how='left')
        filtered_positive['count'] = filtered_positive['count'].fillna(0)
        data.append(go.Scatter(x=filtered_positive['date'], y=filtered_positive['count'], name='Positive Sentiment', fill='tozeroy', line=dict(color=colors['positive'])))

    if 'negative' in sentiment:
        filtered_negative = df[df['sentiment'] == 'negative']
        filtered_negative = filtered_negative.groupby(['date']).size().reset_index(name='count')
        filtered_negative = filtered_negative[(filtered_negative['date'] >= start_date) & (filtered_negative['date'] <= end_date)]
        filtered_negative = pd.merge(date_range.to_frame(name='date'), filtered_negative, on='date', how='left')
        filtered_negative['count'] = filtered_negative['count'].fillna(0)
        data.append(go.Scatter(x=filtered_negative['date'], y=filtered_negative['count'], name='Negative Sentiment', fill='tozeroy', line=dict(color=colors['negative'])))
        
    if 'all' in sentiment:
        filtered_negative = df
        filtered_negative = filtered_negative.groupby(['date']).size().reset_index(name='count')
        filtered_negative = filtered_negative[(filtered_negative['date'] >= start_date) & (filtered_negative['date'] <= end_date)]
        filtered_negative = pd.merge(date_range.to_frame(name='date'), filtered_negative, on='date', how='left')
        filtered_negative['count'] = filtered_negative['count'].fillna(0)
        data.append(go.Scatter(x=filtered_negative['date'], y=filtered_negative['count'], name='All Sentiment', fill='tozeroy', line=dict(color=colors['all'])))

    layout = go.Layout(title=title, xaxis={'title': 'Date'}, yaxis={'title': 'Sentiment Count'})
    


 
    #df = df.drop(df.columns[1], axis=1)
    if selected_file is not None:
        # Load the updated CSV file
        
        df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))
        df.to_csv('Datasets/classified_data.csv', index=False)
     
        # Calculate the sentiment values
        total_sentiment = len(df)
        total_sentiment = "{:,}".format(total_sentiment)
        #Total sentiments from the dataframe
        positive_sentiment = len(df[df['sentiment'] == 'positive'])
        positive_sentiment = "{:,}".format(positive_sentiment)
        negative_sentiment = len(df[df['sentiment'] == 'negative'])
        negative_sentiment= "{:,}".format(negative_sentiment)
    
        mindate = str(df['created_time'].min())[0:4]
        maxdate = str(df['created_time'].max())[0:4]
        year_range = str(mindate + ' - ' + maxdate)
        
        # Create data for pie chart
        sentiment_count = df['sentiment'].value_counts()
        sentiment_count = sentiment_count.rename(index={'negative': 'negative', 'positive': 'positive'})
        sentiment_count = sentiment_count.rename_axis('sentiment').reset_index(name='count')

        labels = sentiment_count['sentiment']
        values = sentiment_count['count']

        # Create pie chart
        Piefig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker=dict(colors=piecolors))])
        
        # Use pandas' value_counts function to get the word frequencies, excluding the stop words
        word_freq = df["preprocessed_comments"].apply(lambda x: [word for word in x.split() if word not in stop_words]).explode().value_counts()[:10]

        # Convert the resulting Series back to a DataFrame and sort in descending order
        word_freq_df = word_freq.to_frame().reset_index()
        word_freq_df.columns = ["word", "frequency"]
        word_freq_df = word_freq_df.sort_values(by="frequency", ascending=True)
        
        # convert created_time column to datetime
        df['created_time'] = pd.to_datetime(df['created_time'])

        # count the number of values in each date
        ds = df['created_time'].value_counts().reset_index()
        ds.columns = ['date', 'count']
        sorted(ds['date'])
        dates = pd.date_range(start=ds['date'].min() ,end=ds['date'].max(), freq='D')
        ds = pd.DataFrame({'date': ds['date'], 'count': ds['count']})


        def unixTimeMillis(dt):
            ''' Convert datetime to unix timestamp '''
            return int(dt.timestamp())

        def unixToDatetime(unix):
            ''' Convert unix timestamp to datetime. '''
            return pd.to_datetime(unix, unit='s')

        def getMarks(dates, Nth=100):
            ''' Returns the marks for labeling. 
                Every Nth value will be used.
            '''
            result = {}
            years = pd.DatetimeIndex(dates).year.unique()  # Get unique years from dates
            for i, year in enumerate(years):
                date = pd.to_datetime(f'{year}')  # Use January 1st of each year as mark
                result[unixTimeMillis(date)] = str(date.strftime('%Y-%m-%d'))
            return result

        senti = df['sentiment'].value_counts().reset_index()
        senti.columns = ['sentiment', 'count']
        df['created_time'] = pd.to_datetime(df['created_time'])

        ###Sentiment Timeline
        #Converting 'positive' as positive and 'negative' as negative 
        #df['sentiment'] = df['sentiment'].apply(lambda x: 'positive' if x == 'positive' else 'negative')
        positive_df = df[df['sentiment'] == 'positive']
        negative_df = df[df['sentiment'] == 'negative']

        positive_grouped = positive_df.groupby([ds['date']]).size().reset_index(name='count')
        negative_grouped = negative_df.groupby([ds['date']]).size().reset_index(name='count')

        # count the number of values in each date
        counts = df['created_time'].value_counts().reset_index()
        counts.columns = ['date', 'count']

        #Get the time frame of data
        dates = sorted(counts['date'])
        start = str(dates[0])[:4]
        end = str(dates[len(dates)-1])[:4]
        duration = start + ' - ' + end




        #Timeline Graph
        df['date'] = pd.to_datetime(df['created_time'])

        # Define colors for each sentiment
        colors = {'positive': 'green', 'negative': 'red', 'all': '#23adfd'}
        #df = df.drop(df.columns[1], axis=1)

        df = df.drop('Unnamed: 0.1', axis=1, errors='ignore')


        # convert created_time column to datetime
        df['created_time'] = pd.to_datetime(df['created_time'])


        # count the number of values in each date
        ds = df['created_time'].value_counts().reset_index()
        ds.columns = ['date', 'count']
        sorted(ds['date'])
        dates = pd.date_range(start=ds['date'].min() ,end=ds['date'].max(), freq='D')
        ds = pd.DataFrame({'date': ds['date'], 'count': ds['count']})


        
        #Timeline Graph
        df['date'] = pd.to_datetime(df['created_time'])

        # Set the range slider bounds and marks based on the date range in the CSV file
        min_date = df["date"].min().timestamp()
        max_date = df["date"].max().timestamp()
        marks = {int(pd.Timestamp(date).timestamp()): {"label": pd.Timestamp(date).strftime("%b %Y"), "style": {"transform": "rotate(45deg)"}} for date in pd.date_range(start=df["date"].min().date(), end=df["date"].max().date(), freq="M").unique()}
        
        # Set the initial range slider value to the full date range of the CSV file
        value = [min_date, max_date]

        # Define colors for each sentiment
        colors = {'positive': 'green', 'negative': 'red', 'all': '#23adfd'}

        def unixTimeMillis(dt):
            ''' Convert datetime to unix timestamp '''
            return int(dt.timestamp())

        def unixToDatetime(unix):
            ''' Convert unix timestamp to datetime. '''
            return pd.to_datetime(unix, unit='s')

        def getMarks(dates, Nth=100):
            ''' Returns the marks for labeling. 
                Every Nth value will be used.
            '''
            result = {}
            years = pd.DatetimeIndex(dates).year.unique()  # Get unique years from dates
            for i, year in enumerate(years):
                date = pd.to_datetime(f'{year}')  # Use January 1st of each year as mark
                result[unixTimeMillis(date)] = str(date.strftime('%Y-%m-%d'))
            return result

        senti = df['sentiment'].value_counts().reset_index()
        senti.columns = ['sentiment', 'count']
        df['created_time'] = pd.to_datetime(df['created_time'])

        ###Sentiment Timeline
        #Converting 'positive' as positive and 'negative' as negative 
        #df['sentiment'] = df['sentiment'].apply(lambda x: 'positive' if x == 'positive' else 'negative')
        positive_df = df[df['sentiment'] == 'positive']
        negative_df = df[df['sentiment'] == 'negative']

        positive_grouped = positive_df.groupby([ds['date']]).size().reset_index(name='count')
        negative_grouped = negative_df.groupby([ds['date']]).size().reset_index(name='count')

        # count the number of values in each date
        counts = df['created_time'].value_counts().reset_index()
        counts.columns = ['date', 'count']

        #Get the time frame of data
        dates = sorted(counts['date'])
        start = str(dates[0])[:4]
        end = str(dates[len(dates)-1])[:4]
        duration = start + ' - ' + end


        start_date = pd.to_datetime(timestamp_range[0], unit='s')
        end_date = pd.to_datetime(timestamp_range[1], unit='s')
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        title = f'Timeline of Sentiment Counts from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'

        if 'positive' in sentiment:
            filtered_positive = df[df['sentiment'] == 'positive']
            filtered_positive = filtered_positive.groupby(['date']).size().reset_index(name='count')
            filtered_positive = filtered_positive[(filtered_positive['date'] >= start_date) & (filtered_positive['date'] <= end_date)]
            filtered_positive = pd.merge(date_range.to_frame(name='date'), filtered_positive, on='date', how='left')
            filtered_positive['count'] = filtered_positive['count'].fillna(0)
            data.append(go.Scatter(x=filtered_positive['date'], y=filtered_positive['count'], name='Positive Sentiment', fill='tozeroy', line=dict(color=colors['positive'])))

        if 'negative' in sentiment:
            filtered_negative = df[df['sentiment'] == 'negative']
            filtered_negative = filtered_negative.groupby(['date']).size().reset_index(name='count')
            filtered_negative = filtered_negative[(filtered_negative['date'] >= start_date) & (filtered_negative['date'] <= end_date)]
            filtered_negative = pd.merge(date_range.to_frame(name='date'), filtered_negative, on='date', how='left')
            filtered_negative['count'] = filtered_negative['count'].fillna(0)
            data.append(go.Scatter(x=filtered_negative['date'], y=filtered_negative['count'], name='Negative Sentiment', fill='tozeroy', line=dict(color=colors['negative'])))
        
        if 'all' in sentiment:
            filtered_negative = df
            filtered_negative = filtered_negative.groupby(['date']).size().reset_index(name='count')
            filtered_negative = filtered_negative[(filtered_negative['date'] >= start_date) & (filtered_negative['date'] <= end_date)]
            filtered_negative = pd.merge(date_range.to_frame(name='date'), filtered_negative, on='date', how='left')
            filtered_negative['count'] = filtered_negative['count'].fillna(0)
            data.append(go.Scatter(x=filtered_negative['date'], y=filtered_negative['count'], name='All Sentiment', fill='tozeroy', line=dict(color=colors['all'])))
        layout = go.Layout(title=title, xaxis={'title': 'Date'}, yaxis={'title': 'Sentiment Count'})
    
       
    import plotly.colors as colors

    # Define a custom color scale that goes from blue to black
    blue_black = colors.make_colorscale(["#002B69","#0068B0","#008AD6", "#23adfd"])

        
    # Create a bar graph using Plotly Express with the custom color scale
    freqFig = px.bar(
        word_freq_df, x="frequency", y="word",
        orientation="h",
        text="frequency",
        color="frequency",
        color_continuous_scale=blue_black,
    )

    # Modify the layout of the figure to remove the title and axis titles
    freqFig.update_layout(
        title='Most frequent words',
        xaxis=dict(showgrid=False, showticklabels=True),
        yaxis=dict(showgrid=False, showticklabels=True),
        plot_bgcolor='white',
        height=400,  # set the height to 400 pixels
        width=400,    # set the width to 600 pixels
        margin=dict(l=50, b=125),
        title_x=0.5,  # move the title 50% to the right
    )
    freqFig.update_traces(textfont_color='white')
    
    top_words = get_top_words(label_value)
    wc = WordCloud(background_color='white', width=900, height=230, colormap=BB )
    wc.generate_from_frequencies(frequencies=top_words)
    wc_image = wc.to_image()
    img = BytesIO()
    wc_image.save(img, format='PNG')
    encoded_image = base64.b64encode(img.getvalue()).decode()
    
    wordcloudimg = 'data:image/png;base64,{}'.format(encoded_image)
    
    
    graph_data = {'data': data, 'layout': layout}    
    return [total_sentiment, positive_sentiment, negative_sentiment, year_range, Piefig, freqFig, wordcloudimg,  graph_data]



@callback(
    [Output("date-slider", "min"),
     Output("date-slider", "max"),
     Output("date-slider", "value"),
     Output("date-slider", "marks")],
    [Input('file-dropdown', 'value')])
def update_range(contents):
    
    df = pd.read_csv('Datasets/classified_data.csv')
    df = pd.DataFrame(df)
    df['created_time'] = pd.to_datetime(df['created_time'])

    df['date'] = pd.to_datetime(df['created_time'])

    # Set the range slider bounds and marks based on the date range in the CSV file
    min_date = df["date"].min().timestamp()
    max_date = df["date"].max().timestamp()
    marks = {int(pd.Timestamp(date).timestamp()): {"label": pd.Timestamp(date).strftime("%b %Y"), "style": {"transform": "rotate(45deg)"}} for date in pd.date_range(start=df["date"].min().date(), end=df["date"].max().date(), freq="M").unique()}
    
    # Set the initial range slider value to the full date range of the CSV file
    value = [min_date, max_date]
    

    if contents is not None:
        # Parse the CSV file
        df = pd.read_csv(os.path.join(DATA_FOLDER, contents))
        df['created_time'] = pd.to_datetime(df['created_time'])


        df['date'] = pd.to_datetime(df['created_time'])

        # Set the range slider bounds and marks based on the date range in the CSV file
        min_date = df["date"].min().timestamp()
        max_date = df["date"].max().timestamp()
        marks = {int(pd.Timestamp(date).timestamp()): {"label": pd.Timestamp(date).strftime("%b %Y"), "style": {"transform": "rotate(45deg)"}} for date in pd.date_range(start=df["date"].min().date(), end=df["date"].max().date(), freq="M").unique()}
        
        # Set the initial range slider value to the full date range of the CSV file
        value = [min_date, max_date]
        
        return min_date, max_date, value, marks
     # If contents is None, return default values
    return min_date, max_date, value, marks

