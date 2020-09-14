import pandas as pd
import numpy as np

import dash

dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go

# Import for data methods
import sys

sys.path.insert(0, '..')
from data.get_data import get_johns_hopkins
from data.process_JH_data import store_relational_JH_data
from features.build_features import features_generator

from scipy import optimize
from scipy import integrate

from datetime import datetime

# Get the latest COVID data from github and do the necessary preprocessing
get_johns_hopkins()
store_relational_JH_data()
features_generator()

import os

print(os.getcwd())
df_input_large = pd.read_csv('../../data/processed/COVID_final_set.csv', sep=';')

# Dataset of all countries
df_analyse = pd.read_csv('../../data/processed/COVID_final_set.csv', sep=';')

# The population of all countries are stored in population.csv file. Read it.
df_population = pd.read_csv('../../data/processed/population.csv', sep=';')


def SIR_model_t(SIR, t, beta, gamma):
    """ Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta:

        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)

    """
    S, I, R = SIR
    dS_dt = -beta * S * I / N0
    dI_dt = beta * S * I / N0 - gamma * I
    dR_dt = gamma * I
    return dS_dt, dI_dt, dR_dt


def SIR_model(y_data, population):
    """
    SIR model helper function
    :param y_data: truth data
    :param population: population of the country
    :return: the fitted result
    """
    global SIR, t, N0

    ydata = np.array(y_data)
    t = np.arange(len(ydata))

    # Initialize the appropriate percent of population that are susceptible for infection
    si = 0.2

    N0 = si * population
    I0 = ydata[0]
    S0 = N0 - I0
    R0 = 0
    SIR = (S0, I0, R0)

    # initialize beta and gamma
    popt = [0.4, 0.1]

    fit_odeint(t, *popt)

    popt, pcov = optimize.curve_fit(fit_odeint, t, ydata, bounds=(0, [0.5, 0.2]))
    perr = np.sqrt(np.diag(pcov))

    print('standard deviation errors : ', str(perr), ' start infect:', ydata[0])
    print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

    fitted = fit_odeint(t, *popt)

    return t, fitted


def fit_odeint(x, beta, gamma):
    return integrate.odeint(SIR_model_t, SIR, t, args=(beta, gamma))[:, 1]


fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Enterprise Data Science - COVID-19 data

    This Project implements automated data gathering, data transformations,
    filtering and machine learning to approximating the doubling time, and
    (static) deployment of responsive dashboard of COVID-19 data for multiple
    countries.

    '''),

    dcc.Tabs(id='main_tab', value='main_tab', children=[
        dcc.Tab(id='tab1', label='COVID cases visualization based on countries', value='tab1', children=[

            dcc.Markdown('''
    ## Select countries for visualization
    '''),

            dcc.Dropdown(
                id='country_drop_down',
                options=[{'label': each, 'value': each} for each in df_input_large['country'].unique()],
                value=['Germany', 'India', 'Netherlands'],  # which are pre-selected
                multi=True
            ),

            dcc.Markdown('''
        ## Select confirmed COVID-19 cases or the approximated doubling time
        '''),

            dcc.Dropdown(
                id='doubling_time',
                options=[
                    {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
                    {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
                    {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
                    {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
                ],
                value='confirmed',
                multi=False
            ),

            dcc.Graph(figure=fig, id='main_window_slope')
        ]),
        dcc.Tab(id='tab2', label='SIR Model of COVID Infection spread', value='tab2', children=[
            dcc.Markdown('''

    # SIR model fitting of COVID-19 spread on the second tab

    '''),

            dcc.Dropdown(
                id='select_country',
                options=[{'label': each, 'value': each} for each in df_analyse['country'].unique()],
                value='India',  # which are pre-selected
                multi=False),

            dcc.Graph(figure=fig, id='SIR_figure')
        ])

    ])

])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
     Input('doubling_time', 'value')])
def update_figure(country_list, show_doubling):

    if 'doubling_rate' in show_doubling:
        my_yaxis = {'type': "log",
                    'title': 'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
                    }
    else:
        my_yaxis = {'type': "log",
                    'title': 'Confirmed infected people (source johns hopkins csse, log-scale)'
                    }

    traces = []
    for each in country_list:

        df_plot = df_input_large[df_input_large['country'] == each]

        if show_doubling == 'doubling_rate_filtered':
            df_plot = df_plot[
                ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                 'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
        else:
            df_plot = df_plot[
                ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()

        traces.append(dict(x=df_plot.date,
                           y=df_plot[show_doubling],
                           mode='markers+lines',
                           opacity=0.9,
                           name=each
                           )
                      )

    return {
        'data': traces,
        'layout': dict(
            width=1280,
            height=720,

            xaxis={'title': 'Timeline',
                   'tickangle': -45,
                   'nticks': 20,
                   'tickfont': dict(size=14, color="#7f7f7f"),
                   },

            yaxis=my_yaxis
        )
    }


# SIR modelling call back
@app.callback(
    Output('SIR_figure', 'figure'),
    [Input('select_country', 'value')])
def update_SIR_figure(select_country):
    traces = []

    df_plot = df_analyse[df_analyse['country'] == select_country]
    df_plot = df_plot[['state', 'country', 'confirmed', 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()
    df_plot.sort_values('date', ascending=True).head()

    # choosing a more valid date to start the plotting
    df_plot = df_plot.confirmed[65:]

    population = df_population[df_population['COUNTRY'] == select_country]['Value'].values[0]

    t, fitted = SIR_model(df_plot, population)

    traces.append(dict(x=t,
                       y=fitted,
                       mode='markers',
                       opacity=0.9,
                       name='SIR-fitted')
                  )

    traces.append(dict(x=t,
                       y=df_plot,
                       mode='lines',
                       opacity=0.9,
                       name='Original Data')
                  )

    return {
        'data': traces,
        'layout': dict(
            width=1280,
            height=720,
            title='SIR model fitting for ' + select_country,

            xaxis={'title': 'Days',
                   'tickangle': -45,
                   'nticks': 20,
                   'tickfont': dict(size=14, color="#7f7f7f"),
                   },

            yaxis={'title': "Infected population"}
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
