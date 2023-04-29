import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import dash
from dash import dcc
from dash import html
import pickle
from dash.dependencies import Input, Output, State
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import base64
import io
import dash_bootstrap_components as dbc
import os
import sys
import module


dash.register_page(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'])

layout = html.Div(className = '', children=[
    
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
        ]),
    ]),
    
    
    

    html.Div([

    html.Div(id='import_div3', className='import_div', children=[
    html.Div(id='', className='', children=[
         html.Label(id='info-icon', children='Import CSV file Here:'),
        dcc.Upload(
        id='upload-data',
        style={
            'width': '100%',
            'height': '150px',
            'lineHeight': '150px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'cursor':'pointer',

        },
        multiple=False,
    ), 
    ],title="The first step is to import the CSV file that contains the text data. Users can either drag and drop the file or select it from their device."),
   
        html.Br(id='', className='', children=[]),
        html.Div(
        html.Span(
            "Model Selection:",
            className="info-icon",
            title="Users should select SVM or Naive bayes for classification from the given options."
        ),
        className="info-container"
        ),
        dcc.Dropdown(
        id='model-selector',
        options=[
            {'label': 'Support Vector Machine', 'value': 'svm'},
            {'label': 'Naive Bayes', 'value': 'nb'}
        ],
        value='svm'
        ),
        
        
        html.Br(id='', className='', children=[]),
        
        html.Div(
        html.Span(
            'Select Text Column:',
            className="info-icon",
            title="Users should select the column that contains the text data which will be used for classification."
        ),
        className="info-container"
        ),
        dcc.Dropdown(
            id='text-col-dropdown'
        ),
        html.Div(
        html.Span(
            'Select Date Column:',
            className="info-icon",
            title="If the CSV file contains a date column, users should select it."
        ),
        className="info-container"
        ),
        dcc.Dropdown(
            id='date-col-dropdown'
        ),
        
        html.Br(id='', className='', children=[]),
       
        html.Div(
        html.Span(
            'Merge File with:',
            className="info-icon",
            title="If you want to merge the CSV file with another file, users can select the merge file option"
        ),
        className="info-container"
        ),
        dcc.Dropdown(
            id='dropdown-file2',
            options=[],
            value=None
        ),
       
        
        html.Br(id='', className='', children=[]),
        html.Div(id='', className='', children=[
        html.Button('Preprocess and Classify',className='hover-btn', id='save-csv-button',style={
                                                        'border': '1px solid grey',
                                                        'borderRadius': '10px',
                                                        'color': 'black',
                                                        'padding': '8px 16px',
                                                        'width':'40%'}),
        ],style={
        'display': 'flex',
        'justifyContent': 'center'}),
        
        
        
        html.Br(id='', className='', children=[]),
        
         
        html.Div(id='', className='border', children=[
            html.Div(id='output-data-upload'),
            html.Div(id='output-data-merge'),
            html.Br(id='', className='', children=[]),
            html.Div(id='', className='', children=[
                html.Div(id='', className='', children=[
                    
                ],style={'margin':'20px'}),
                dcc.Loading(
            id="loading",
            children=[
                html.Div(id='terminal-output',style={'zoom':'130%'}),
            ],
            type="circle",style={'padding-top':'450px'}
            ),

            ],style={'height':'640px','padding-left':'30px','backgroundColor': 'white','border': '1px solid grey','overflow':'hidden'}),
            
        ],style={'zoom':'85%'})
    
    ]),
    
    html.Div(id='import_div4', className='import_div', children=[
        html.Div(id='', className='', children=[
            html.H2(id='', className='', children='Instruction Manual',style={'text-align':'center'}),
            html.Div(id='', className='', children=[
            html.Label(id='', className='', children='Step 1:  ',style={'font-weight':'bold'}),
            html.Label(id='', className='', children=' Import your CSV file.'),
            html.Br(id='', className='', children=[]),
        
            html.Br(id='', className='', children=[]),
            html.Br(id='', className='', children=[]),
            html.Label(id='', className='', children='Step 2:  ',style={'font-weight':'bold'}),
            html.Label(id='', className='', children=' Select model.'),
            html.Br(id='', className='', children=[]),
            html.Br(id='', className='', children=[]),
            
            
            html.Table([
            html.Thead(
                html.Tr([html.Th('Support Vector Machine',style={'padding': '10px'}),html.Th('Naive Bayes',style={'padding': '10px'})],
                        style={'background-color': '#bbe6ff', 'font-weight': 'bold'})
            ),
            html.Tbody([
                html.Tr([
                    html.Td('SVM is a linear classification algorithm that finds the hyperplane that best separates the data into different classes by maximizing the margin between the hyperplane and the closest points from each class.', style={'padding': '10px'}), 
                    html.Td('Naive Bayes is a probabilistic classification algorithm that makes predictions based on Bayes theorem and the assumption that the features are independent of each other, given the class label, by calculating the probability of each class given the features of the input data and choosing the class with the highest probability.', style={'padding': '10px'}), 
                ], style={'background-color': '','height':'100px'}),
                
            ], style={'border-collapse': 'collapse'})
        ], style={'border': '1px solid gray','width':'100%','table-layout': 'fixed'})
        ],style={'margin':'20px'}),
            
            
        html.Div(id='', className='', children=[
            html.Label(id='', className='', children='Try the models here!  ',style={'font-weight':'bold'}),
            html.Br(id='', className='', children=[]),
            html.Br(id='', className='', children=[]),
            html.Label('Model selection for input text:'),
            
            dcc.Dropdown(
            id='text-input-model',
            options=[
                {'label': 'Support Vector Machine', 'value': 'svm'},
                {'label': 'Naive Bayes', 'value': 'nb'}
            ],
            value='svm'
            ),
        ],style={'margin':'20px'}),
        
        html.H5('Input text for sentiment analysis', style={'text-align': 'center','font-size':'15px'}),
        html.Div([
        dcc.Input(
            id='text-input',
            placeholder='Enter a value...',
            type='text',
            value='',
            style={'text-align': 'center','width': '50%', }
        ),
        ], style={'display': 'flex', 'justify-content': 'center'}),
        html.Br(id='', className='', children=[]),
        dcc.Loading(id='loading-output', type='dot', color='#23adfd', children=[  
                html.Div(id='output-text', style={'text-align': 'center'}),
                dcc.Store(id='loading-state'),
                ],style={}),
        html.Div(id='', className='', children=[
            
            html.Label(id='', className='', children='Step 3: ' ,style={'font-weight':'bold'}),
            html.Label(id='', className='', children=' Assign columns'),
            html.Br(id='', className='', children=[]),
            html.Br(id='', className='', children=[]),
            html.Table([
            html.Thead(
                html.Tr([html.Th('Text Column',style={'padding': '10px'}),html.Th('Date Column',style={'padding': '10px'})],
                        style={'background-color': '#bbe6ff', 'font-weight': 'bold'})
            ),
            html.Tbody([
                html.Tr([
                    html.Td('Contains raw comments or text ("string")', style={'padding': '5px'}), 
                    html.Td('Contains dates of comments (YY-MM-DD)', style={'padding': '5px'}), 
                ], style={'background-color': '','height':'10px'}),
                
            ], style={'border-collapse': 'collapse'})
            ], style={'border': '1px solid gray','width':'100%','table-layout': 'fixed'}),
            html.Br(id='', className='', children=[]),
            html.Br(id='', className='', children=[]),
            html.Label(id='', className='', children='Step 4:  ',style={'font-weight':'bold'}),
            html.Label(id='', className='', children='Please select a file for merging the new data, or leave the field blank if there is no need for merging'),
            html.Br(id='', className='', children=[]),
            html.Br(id='', className='', children=[]),
            html.Label(id='', className='', children='Step 5:  ',style={'font-weight':'bold'}),
            html.Label(id='', className='', children='Click " Preprocess and Classify" button.'),
            html.Br(id='', className='', children=[]),
            html.Br(id='', className='', children=[]),
            html.Label(id='', className='', children='Step 6:  ',style={'font-weight':'bold'}),
            html.Label(id='', className='', children=' Proceed to Dashboard or issues to view the results.'),
            html.Br(id='', className='', children=[]),
            html.Br(id='', className='', children=[]),
        ],style={'margin':'20px','padding-top':'100px'}),
        ],style={'backgroundColor': 'white','border': '1px solid grey','overflow':'hidden','height':'1050px',}),
        
     
    ]),
       
    ],style={'zoom':'85%'}),
   

], style ={'width':'100%'})

        
import subprocess
from dash.exceptions import PreventUpdate
import input_preprocessing


@callback(
    Output('output-text', 'children'),
    [Input('text-input', 'value')],
    [State('loading-state', 'data'),State('text-input-model', 'value')]
)
def predict_sentiment(text, loading_state,textInputModel):
                     
    if not text:
        return ''
    else:
        text = str(text)
        if textInputModel == "nb":
            # Load the saved SVM model from file
            with open('Models/nb_model.pkl', 'rb') as f:
                mdl = pickle.load(f)

            with open('Models/nb_vectorizer.pkl', 'rb') as f:
                vec = pickle.load(f)
        else:
            with open('Models/svm_model.pkl', 'rb') as f:
                mdl = pickle.load(f)

            with open('Models/svm_vectorizer.pkl', 'rb') as f:
                vec = pickle.load(f)
        # Apply trained model to input text
        result = input_preprocessing.process_input(text)
        result = [result]
        input_data = np.array(result)
        input_tfidf = vec.transform(input_data)
        pred = mdl.predict(input_tfidf)
        # Display predicted sentiment
        output = html.P([
            html.H5('Predicted sentiment: {}'.format(pred[0])),
        ])
        
       
        return output

@callback(
    Output('loading-state', 'data'),
    [Input('text-input', 'value')],
    [State('loading-state', 'data')]
)
def update_loading_state(text, loading_state):
    if not text:
        raise PreventUpdate
    return True
        
############################
# Define a function to read in the uploaded CSV file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8'))
            )
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df




# Define callback to populate dropdown options when CSV is uploaded
@callback(Output('text-col-dropdown', 'options'),
              Output('date-col-dropdown', 'options'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'))
def update_dropdown_options(contents, filename):
    if contents:
        df = parse_contents(contents, filename)
        # Populate dropdown options with column names from CSV
        date_col_options = [{'label': col, 'value': col} for col in df.columns if df[col].dtype == 'object']
        text_col_options = [{'label': col, 'value': col} for col in df.columns if df[col].dtype == 'object']
        
        return date_col_options, text_col_options
    else:
        return [], []


# Define callback to save selected columns to CSV
@callback(
    Output('output-data-upload', 'children'),
    Output('terminal-output', 'children'),
    Output('output-data-merge', 'children'),
    Input('save-csv-button', 'n_clicks'),
    State('text-col-dropdown', 'value'),
    State('date-col-dropdown', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('model-selector', 'value'),
    State('dropdown-file2', 'value')
)
def process_data(n_clicks, text_col, date_col, contents, filename, selectedModel,file2):
    pred = ''
    if n_clicks is not None:
        df = parse_contents(contents, filename)
        selected_cols = [text_col]
        if date_col is not None:
            selected_cols.append(date_col)
        df_selected = df[selected_cols]
        df_selected = df_selected.rename(columns={date_col: 'created_time', text_col: 'raw_comments'})
        df_selected.to_csv('Datasets/raw_data.csv', index=False, encoding='utf-8')
        
        output = []
        commands = ['py preprocessor.py']
        for command in commands:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                output.append(line.decode('utf-8').strip())
        if selectedModel == "nb":
            model = "nb"
        else:
            model = "svm"
        pred = predictor(df,model,file2)
        output.append('Classifying Data...')
        output.append('Model:')
        output.append(model)
        output.append('Done!')
        print(pred)
        output = '\n'.join(output)
        return 'CSV file imported!', dcc.Markdown(f'```{output}```'),pred

    else:
        return '', '',pred,


import datetime

@callback(Output('upload-data', 'children'),
              Input('upload-data', 'filename'))
def update_upload_component(filename):
    if filename is not None:
        return html.Div([
            html.I(className='fa fa-file-csv',style={'font-size': '30px'}),
            html.Span(filename)
        ])
    else:
        return html.Div([
            html.I(className='fas fa-file-upload',style={'font-size': '30px'}),
            '    Drag and Drop or ',
            html.A('Select CSV File'),
            
        ],style={'color':'gray'})
        
        
import csv  
def predictor(df,mod, file2):
    # Load the saved SVM model from file
    if mod == 'nb':
        with open('Models/nb_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('Models/nb_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        with open('Models/svm_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('Models/svm_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
    # Read preprocessed data from a CSV file into a Pandas DataFrame and drop NaN values
    df = pd.read_csv('Datasets/preprocessed_data.csv').dropna()

    # Apply trained model to test data
    test_tfidf =  vectorizer.transform(df['preprocessed_comments'])
    pred = model.predict(test_tfidf)
    df['sentiment'] = pred 

    # Save and merge the DataFrame to a CSV file
    print(mod)
    
    # Check if both files were selected
    if file2 is not None:
        # Get the directory path of the csv files
        folder_path = 'Datasets/classified/'
        
        # Read the uploaded file as a pandas dataframe
        file1 = df
        # Read the csv file for file 2 as a pandas dataframe
        df2 = pd.read_csv(os.path.join(folder_path, file2))
        
        # Merge the dataframes
        merged_df = pd.concat([file1, df2])
        
        # Find the minimum and maximum dates in the "date" column of the dataframes
        min_date = merged_df['created_time'].min()
        max_date = merged_df['created_time'].max()
        
       
        
        # Write the merged dataframe to a new csv file
        now = datetime.datetime.now()
        merged_filename = mod + '_' +'merged_' + str(min_date) + '_' + str(max_date) + '_'  + now.strftime("%H-%M-%S") + '.csv'
        merged_df.to_csv(os.path.join(folder_path, merged_filename), index=False)
        
        # Return a message with the name of the merged file
        return html.Div([
            'File saved as ',
            html.A(merged_filename, href=os.path.join(folder_path, merged_filename))
        ])
    else:
        
        folder_path = 'Datasets/classified/'
        
        # Find the minimum and maximum dates in the "date" column of the dataframes
        min_date = df['created_time'].min()
        max_date = df['created_time'].max()
        
       
        
        # Write the merged dataframe to a new csv file
        now = datetime.datetime.now()
        filename = mod + '_' + str(min_date) + '_' + str(max_date) + '_'  + now.strftime("%H-%M-%S") + '.csv'
        df.to_csv(os.path.join(folder_path, filename), index=False)
        
        # Return a message with the name of the merged file
        return html.Div([
            'Files merged and saved as ',
            html.A(filename, href=os.path.join(folder_path, filename))
        ])
        
        
@callback(
    Output('dropdown-file2', 'options'),
     Input('save-csv-button', 'n_clicks'),
      Input('model-selector', 'value'),
)
def update_dropdowns(n_clicks,model):
    # Get the directory path of the csv files
    folder_path = 'Datasets/classified/'
    if model == 'nb':
        # Filter for csv files in the folder
        csv_files = [f for f in os.listdir(folder_path) if (f.startswith(('nb')))]
    elif model == 'svm':
        csv_files = [f for f in os.listdir(folder_path) if (f.startswith(('svm')))]
        
    # Create a list of dropdown options
    dropdown_options = [{'label': file, 'value': file} for file in csv_files]
    
    # Return the options for file 2 only
    return dropdown_options

