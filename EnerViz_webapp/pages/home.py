import dash
from dash import html, dcc

dash.register_page(__name__, path='/')

layout = html.Div( 
  
    className='body home',id ='home',children=[
    html.Div(id='home_icon', className='', children=[
        
               
    html.Div(id='', className='main-container', children=[
    dcc.Link( children=[
         html.Div(id='', className='card_about animated-element element1', children=[
            html.Img(src= '/assets/logo.png',  className='icon', style={ 'width': '18%'}),
            html.Img(src= '/assets/logo.png',  className='', style={ 'width': '85%', 'position':'absolute','margin':'-30% 0px 0px 42%','opacity':'20%'}),
            html.Span(id='', className='', children=[html.H1(id='', className='animated-text title_about', children='EnerViz')]),
            html.Span(id='', className='subtitle_about', children=['About']),
             html.Span(id='', className='description_about', children=["In this page, a short informational video and paragraph can be viewed by the user to better understand the system."]),    
            ]),
    ], href='/about'),
        html.Div(id='', className='bottom_cards', children=[
            dcc.Link( children=[
            html.Div(id='', className='card animated-element element2', children=[
                html.Img(src= '/assets/dash.png',  className='icon', style={ 'width': '20%'}),
                html.Img(src= '/assets/dash.png',  className='', style={ 'width': '70%', 'position':'absolute','margin':'-250px 0px 0px 240px','opacity':'20%'}),
                html.Span(id='', className='title', children=[html.H2(id='', className='', children='Dashboard')]),
                html.Span(id='', className='subtitle', children=['About']),
                html.Span(id='', className='description', children=["This page provides the overall reflection and visualizations of the results from the model training conducted by the user. It contains graphs, word map, and charts that depict the information and patterns that were extracted from the dataset."]),    
            ]),
            ], href='/dashboard'),
            dcc.Link( children=[
            html.Div(id='', className='card animated-element element3', children=[
                html.Img(src= '/assets/issue.png',  className='icon', style={ 'width': '20%'}),
                html.Img(src= '/assets/issue.png',  className='', style={ 'width': '70%', 'position':'absolute','margin':'-250px 0px 0px 240px','opacity':'20%'}),
                html.Span(id='', className='title', children=[html.H2(id='', className='', children='Issues')]),
                html.Span(id='', className='subtitle', children=['About']),
                html.Span(id='', className='description', children=["The purpose of this page is to highlight the certain issues that are the topic of most of the negative sentiments. The trend of the appearance of such issues can be viewed on the line graph and the comments that are related to the said issue are listed below."]),    
            ]),
            ], href='/issues'),
        ]),
    ]),
   ]), 

    

    html.H3(id='', className='footer', children='EnerViz',style={})
    
    
],style ={})