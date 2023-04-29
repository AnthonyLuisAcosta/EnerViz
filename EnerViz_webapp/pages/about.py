import dash
from dash import html, dcc

dash.register_page(__name__)


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
        ]),
    ]),
    
 
    html.Div(id='', className='', children=[
            
            html.Br(id='', className='', children=[]),
            html.Img(src= '/assets/logo.png',  className='image_about', style={ 'width': '10%'}),
            html.H1(id='', className='', children='EnerViz: Albay Electric Provider Sentiment Visualizer',style={'text-align':'center'}),
            
            html.Video(src='/assets/elecviz_vid.mp4', className='video_about', autoPlay=False, controls=True, style={'width':'50%'}),
            html.H1(id='', className='', children='About',style={'text-align':'center'}),
            html.Div(
            style={
                'textAlign': 'center',
                'width': '50%',
                'margin': 'auto',
            },
            children=[
                html.P(
                    style={
                        'text-align': 'justify',
                    },
                    children=" Welcome to EnerViz - the Albay Electric Provider Sentiment Visualizer! EnerViz is a web application designed to help you visualize and analyze sentiment data related to the Albay Electric Provider. Our goal is to provide a user-friendly platform for users to track and understand the public sentiment towards the Albay Electric Provider. Our sentiment visualizer uses natural language processing and machine learning techniques to analyze comments and tweets to determine the overall sentiment towards the Albay Electric Provider. The data is then visualized in an easy-to-understand format, allowing you to quickly identify trends and gain insights into the public's perception of the Albay Electric Provider. We hope that EnerViz will be a valuable tool for Albay Electric Provider's customers, stakeholders, and management team. Whether you are interested in tracking sentiment related to customer service, rates, or any other aspect of the Albay Electric Provider, our sentiment visualizer can help you stay informed. Thank you for using EnerViz. If you have any questions or feedback, please don't hesitate to contact us.",
                )
            ]
        )

    ],style={'margin':'10px'}),
    
    ],style={
    'font-size': '1em',
    'font-weight': '400',
    'font-family': 'Arial',
    'color': 'rgb(50, 50, 50)',
    'height':'120vh'})

