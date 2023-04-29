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
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import os
from dash import dash_table
import shutil
from collections import Counter
import re


dash.register_page(__name__, refresh=True)
global df
df = pd.read_csv('Datasets/classified_data.csv')
df = pd.DataFrame(df)
#df = df.dropna()

df['preprocessed_comments'] = df['preprocessed_comments'].apply(str)


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

#Get comments based on filters
def filter_comments(df, start_date, end_date, sentiment, word_list):
    # Filter the DataFrame based on the date range and sentiment
    df['created_time'] = pd.to_datetime(df['created_time'])
    filtered_df = df[
        (df['created_time'] >= start_date) &
        (df['created_time'] <= end_date) &
        (df['sentiment'].isin(sentiment))
    ].copy()
    
    # Filter the comments based on the list of words and phrases
    word_set = set(word_list)
    filtered_df['contains_word'] = filtered_df['preprocessed_comments'].str.contains('|'.join(word_set))
    filtered_df['contains_word'] = filtered_df['contains_word'].apply(lambda x: 1 if x else 0)

    return filtered_df



layout = html.Div(className='body', children=[
             dcc.Location(id='url', refresh=True),
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
            html.Img(className='nav-reload',src='/assets/refresh.png',id='refresh-button', n_clicks=0),
           
        ]),
       
    ]),
    
        html.Div(id='', className='', children=[
         
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
        html.Div(id='check-box', className='check-box', children=[
            dcc.Checklist(
            id='sentiments-checkbox',
            options=[
                {'label': 'Positive Sentiment', 'value': 'positive'},
                {'label': 'Negative Sentiment', 'value': 'negative'},
                {'label': 'All Sentiment', 'value': 'all'}
            ],style={'margin':'20px'},
            value=['all'],
            labelStyle={'display': 'inline-block','margin':'10px'}
        ),
                   
        ],style={}),
    html.Div(id='', className='', children=[
            dcc.RangeSlider(
        id='date-range-slider',
        min=df['created_time'].min().timestamp(),
        max=df['created_time'].max().timestamp(),
        value=[df['created_time'].min().timestamp(), df['created_time'].max().timestamp()],
        marks = {int(pd.Timestamp(date).timestamp()): {"label": pd.Timestamp(date).strftime("%b %Y")} for date in pd.date_range(start=df['created_time'].min().date(), end=df['created_time'].max().date(), freq="M").unique()},
        step=None
    ),
    ],style={'width':'90%','margin':'20px'}),


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
    dash.dependencies.Input('date-range-slider', 'value'),
    Input('dropdown2', 'value'),
    Input('dropdown1', 'value'),
    dash.dependencies.Input('sentiments-checkbox', 'value'),
    Input('url', 'pathname'),
    Input('refresh-button', 'n_clicks'),allow_duplicate=True
    
)
def update_graph(date_range, value2, value1, sentiment,path, n_clicks):
    file_path = 'pages/__pycache__'
            
    if os.path.exists(file_path):
        pycache_path_pages = os.path.join(os.getcwd(), "pages/__pycache__")
        shutil.rmtree(pycache_path_pages)
        
    global df
    df = pd.read_csv('Datasets/classified_data.csv')
    df = pd.DataFrame(df)
    #df = df.dropna()

    df['preprocessed_comments'] = df['preprocessed_comments'].apply(str)


    df['created_time'] = pd.to_datetime(df['created_time'])
    
    # Define the keywords 
    option1_allwords = []
    option1_words = ['reading','sala reading', 'maling reading', 'hindi tama', 'mali mali','mali reading','bako tama reading','sarala reading','sarala bill',
                     'sala bill','sarala bill','mali bill','hindi tama bill','sala metro','mali metro','raot metro','sarala metro', 'mali bayadan', 'sarala bayadan',
                     'sala bayadan','raot kontador','sala kontador','mali kontador']
    option1_words1 = ['walang bill', 'tagal bill', 'dumating bill', 'wala bill', 'wara bill', 'wra bill', 'wala bill','delay bill','late bill','haloy bill','haluyon bill']
    option1_words2 = ['mahal bill', 'mahal singil', 'mahalon bayadan', 'mahalon singil', 'increase kwh','mahal bayadan','sobra singil','sobra bayadan','grabe bayadan']

    # Merge the keywords
    option1_allwords.extend(option1_words)
    option1_allwords.extend(option1_words1)
    option1_allwords.extend(option1_words2)

    option2_allwords = []
    #option2_words = ['brownout', 'palsok', 'blackout','abiso', 'advisory', 'interruption', 'supply', 'power', 'wara', 'wra', 
    #                 'wla', 'wala', 'mayo', 'mainitun', 'init', 'kuryente', 'ibalik', 'ilaw', 'negosyo', 'damay', 'lowbat', 
    #                 'pakibalik', 'balik', 'extended', 'schedule', 'off']
    option2_words = ['brownout', 'blackout', 'interruption','wara ilaw','wara kuryente','warang ilaw','warang kuryente','warang power','wara power'
                        ]
    option2_words1 = ['patay','patay sindi','sindi','patay sinde', 'puropalsok', 'palsok', 'palsuk', 'pawala', 'pakibalik', 'balik',
                    'palsok', 'paltok', 'purupaltok', 'wra nanaman', 'wala nanaman', 'wara nanaman ilaw', 'wala nanaman ilaw','kimat', 
                    'wra ilaw','wla ilaw','pawara','maluya kuryente','maluya power','hina kuryente','hina power','hina ilaw','diit brownout','diit brownout ilaw']
    option2_words2 = ['jumper', 'illegal', 'illigal', 'iligal', 'illegal connection']

    option2_allwords.extend(option2_words)
    option2_allwords.extend(option2_words1)
    option2_allwords.extend(option2_words2)

    option3_allwords = []
    option3_words = ['bagyo', 'uran','paros','kidlat','raot panahon','dampog','tagiti']
    option3_words1 = ['repair', 'ayuson', 'transformer', 'unscheduled', 'poste']
    option3_words2 = ['sulo', 'sabog', 'putok','tumba poste','putol linya','kalayo','spark','laad']

    option3_allwords.extend(option3_words)
    option3_allwords.extend(option3_words1)
    option3_allwords.extend(option3_words2)

    defined_words = []

    defined_words.extend(option1_allwords)
    defined_words.extend(option2_allwords)
    defined_words.extend(option3_allwords)

    #Get comments based on filters
    def filter_comments(df, start_date, end_date, sentiment, word_list):
        # Filter the DataFrame based on the date range and sentiment

        df['created_time'] = pd.to_datetime(df['created_time'])
        filtered_df = df[
            (df['created_time'] >= start_date) &
            (df['created_time'] <= end_date) &
            (df['sentiment'].isin(sentiment))
        ].copy()
        
        # Filter the comments based on the list of words and phrases
        word_set = set(word_list)
        filtered_df['contains_word'] = filtered_df['preprocessed_comments'].str.contains('|'.join(word_set))
        filtered_df['contains_word'] = filtered_df['contains_word'].apply(lambda x: 1 if x else 0)

        return filtered_df

    start_date = pd.to_datetime(date_range[0], unit='s')
    end_date = pd.to_datetime(date_range[1], unit='s')
    date_range = pd.date_range(start_date, end_date)

    #########Graph for issues and sub-issues, filtered by sentiment
    data = []
    ####define colors for the graph
    colors = {'positive': 'green', 'negative': 'red', 'all': '#23adfd'}
    ####Main issue 1 with sub issues
    if value1 == 'option1' or not sentiment:

        if value2 == 'option1-1':
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option1_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')

                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option1_words 
                
        elif value2 == 'option1-2':
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option1_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')

                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option1_words 
            
        elif value2 == 'option1-3':
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option1_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')

                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all']))) 
            selected_words = option1_words2
        else:
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option1_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')

                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option1_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option1_allwords 
    ###Main issue 2 with sub issues
    elif value1 == 'option2':

        if value2 == 'option2-1':
            
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option2_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                    
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                    
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option2_words

        elif value2 == 'option2-2':
            
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option2_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                    
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                    
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option2_words1

        elif value2 == 'option2-3':
            
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option2_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                    
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                    
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option2_words2
        else:
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option2_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
            
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                    
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                    
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option2_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
        selected_words = option2_allwords
    ####Main Issue 3 with sub issues
    elif value1 == 'option3':

        if value2 == 'option3-1':
            
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option3_words)
                print(option3_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                    
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                    
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
     
            
        elif value2 == 'option3-2':
            
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option3_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                    
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                    
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words1)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option3_words1
        elif value2 == 'option3-3':
            
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option3_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                    
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                    
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_words2)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option3_words2
        else:
            
            if 'positive' in sentiment:
                sentiment_filter = ['positive']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter, option3_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['positive'])))
                
            if 'negative' in sentiment:
                sentiment_filter = ['negative']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['negative'])))
                    
            if 'all' in sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
                    
            if not sentiment:
                sentiment_filter = ['positive','negative','neutral']
                filtered_df = filter_comments(df, start_date, end_date, sentiment_filter,  option3_allwords)
                comment_count = filtered_df.groupby('created_time')['contains_word'].sum().reset_index(name='count')
                data.append(go.Scatter(x=comment_count['created_time'], y=comment_count['count'], fill='tozeroy', line=dict(color=colors['all'])))
            selected_words = option3_allwords
        
    ###PLOTTING THE ISSUES with selected sentiment
    layout = go.Layout(title='Sentiment Analysis', xaxis=dict(title='Date'), yaxis=dict(title='Word Count'))
    layout.showlegend=False

    # Compute the total count of word counts in the comment list
    total_counts = comment_count["count"].sum()
    
    # Create a list of the comments that contain any of the selected words
    filtered_df['created_time'] = filtered_df['created_time'].dt.strftime('%Y-%m-%d')
    comments_list = filtered_df[['created_time', 'raw_comments']][filtered_df['contains_word'] > 0].values.tolist()
    comments_list = sorted(comments_list, key=lambda x: x[0])

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


    #####################################################UPDATE BUTTON##############################################################

    return {'data': data, 'layout': layout}, table, total_counts_div

@callback(
    [Output("date-range-slider", "min"),
     Output("date-range-slider", "max"),
     Output("date-range-slider", "value"),
     Output("date-range-slider", "marks")],
    [Input('refresh-button', 'n_clicks')])
def update_range(n_clicks):
    DATA_FOLDER = 'Datasets/classified'
    df = pd.read_csv('Datasets/classified_data.csv')
    df = pd.DataFrame(df)
    df['created_time'] = pd.to_datetime(df['created_time'])

    df['date'] = pd.to_datetime(df['created_time'])

    # Set the range slider bounds and marks based on the date range in the CSV file
    min_date = df["date"].min().timestamp()
    max_date = df["date"].max().timestamp()
    marks = {int(pd.Timestamp(date).timestamp()): {"label": pd.Timestamp(date).strftime("%b %Y")} for date in pd.date_range(start=df["date"].min().date(), end=df["date"].max().date(), freq="M").unique()}
    
    # Set the initial range slider value to the full date range of the CSV file
    value = [min_date, max_date]
    if n_clicks > 0:
        
        DATA_FOLDER = 'Datasets/classified'
        df = pd.read_csv('Datasets/classified_data.csv')
        df = pd.DataFrame(df)
        df['created_time'] = pd.to_datetime(df['created_time'])

        df['date'] = pd.to_datetime(df['created_time'])

        # Set the range slider bounds and marks based on the date range in the CSV file
        min_date = df['date'].min().timestamp()
        max_date = df['date'].max().timestamp()
        marks = {int(pd.Timestamp(date).timestamp()): {"label": pd.Timestamp(date).strftime("%b %Y")} for date in pd.date_range(start=df["date"].min().date(), end=df["date"].max().date(), freq="M").unique()}
        
        # Set the initial range slider value to the full date range of the CSV file
        value = [min_date, max_date]
        
        return min_date, max_date, value, marks
     # If contents is None, return default values
    return min_date, max_date, value, marks
