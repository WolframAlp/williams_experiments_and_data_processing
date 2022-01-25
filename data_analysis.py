# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 10:49:15 2021

@author: willi
"""

'''

Process flow.


Data set for current:
    Validate data:
        - Deviation between first 10 seconds and last 10 seconds may not go above 5 % of average
    
    Validate FN model:
        - R-squared value of above 0.97

Data set for variable voltage:
    Validate data:
        - How?
        
    What do we want to see?
    

Analysis
    Emission Coefficient:
        - Compare two sets of data measured at the same point with the same settings

    Spectrum analysis:
        No image method:
            - Split distribution into rings and sum to infinity (less, cause the value will quickly be zero)
            - Calculate the emission coefficient or use the one from data
            - Calculate the electric field


'''


import numpy as np
import pandas as pd
from data_analysis_tools import AnalysisTools
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
import os
pio.renderers.default='svg'

'''
How to analysis


# Select folder with data
base_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE\\1000GHz_020721_current"

# Provide file name
filename = "current.csv"

tools = AnalysisTools(base_dir)
data = tools.get_data(filename, ret=True)
mod = tools.get_linear_fit(data=data, plot=True, position="all", save_plot=True)
'''


def validity_and_FN_mult(folders, base_folder, filename, yaxis="current", plot=False, save=True):
    # Creates an instance of tools which saves the directory of the data
    tools = AnalysisTools()
    return tools.get_mult_FN_fits(folders=folders, filename=filename, base_folder=base_folder, plot=plot, yaxis=yaxis, save=save)


def validity_and_FN_single(base_dir, filename, yaxis="current", save=False):

    # Creates an instance of tools which saves the directory of the data
    tools = AnalysisTools(base_dir)
    
    # Loads data into a pandas
    # The data is also stored in the analysis tools
    data = tools.get_data(filename, ret=True)
    passed = tools.validate_data()
    
    # Creates linear fits for the data
    mod = tools.get_linear_fit(data=data, plot=True, position="all", save_plot=True, yaxis=yaxis)
    
    # Creates a fit for FN model and plots it
    return tools.get_FN_fit(data, save=save, yaxis=yaxis)


def get_basic_data_plots(base_dir, filename, yaxis="current"):
    tools = AnalysisTools(base_dir)
    data = tools.get_data(filename, ret=True)
    return tools.plot_current_as_E(tools.data, yaxis=yaxis)


def get_emcoef(amp_dir, pow_dir, amplification=1e6, title="Emission Coefficient"):
    amp_tools = AnalysisTools(amp_dir)
    pow_tools = AnalysisTools(pow_dir)
    
    amp_data = amp_tools.get_data("\\current.csv", ret=True)
    pow_data = pow_tools.get_data("\\voltage_and_currents.csv", ret=True)

    positions = amp_data.position.unique()

    amp_background = amp_data[amp_data["position"] == positions[0]]["current"].mean()
    amp_curr = np.array([amp_data[amp_data["position"] == pos]["current"].mean() - amp_background for pos in positions[1:]])

    pow_background = pow_data[pow_data["position"] == positions[0]]["current3"].mean()
    pow_curr = np.array([pow_data[pow_data["position"] == pos]["current3"].mean() - pow_background for pos in positions[1:]])

    emcoef = pow_curr / amp_curr / amplification
    Es = (np.cos(positions[1:] * np.pi / 180))**2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Es, y=emcoef))
    fig.update_layout(
                    title = title,
                    xaxis_title="E-field",
                    yaxis_title="C_emission",
                    # legend_title="Legend Title",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="RebeccaPurple"
                    ))
    fig.show()

    return emcoef

def plot_emission_energy_dist(eme_positions, eme_voltages, eme_currents, save_dir = None):
    
    frame = pd.DataFrame(columns=["position", "voltage", "current", "gradCurrent", "gradCurrentNorm"])
    for i,pos in enumerate(eme_positions):
        grad = np.gradient(eme_currents[i])
        gradNorm = grad / max(grad)
        frame = frame.append(pd.DataFrame([[pos,abs(v) - 1500,c,gc,gn] for v,c,gc,gn in zip(eme_voltages, eme_currents[i], grad, gradNorm)],columns=["position", "voltage", "current", "gradCurrent", "gradCurrentNorm"]))
    
    for pos in eme_positions:
        fig = px.scatter(frame[frame["position"] == pos], x="voltage", y="current")
        fig.update_traces(mode='markers+lines')
        fig.show()
    
    fig = px.scatter(frame, x="voltage", y="gradCurrent", color="position")
    fig.show()
    if save_dir is not None:
        fig.write_html(save_dir + "EnergyDistribution.html")
        
    fig = px.scatter(frame, x="voltage", y="gradCurrentNorm", color="position")
    fig.show()
    if save_dir is not None:
        fig.write_html(save_dir + "EnergyDistributionNormalized.html")
    
    return frame


if __name__ == '__main__':
    
    # Provide file name
    amp_filename = "current.csv"
    pow_filename = "voltage_and_currents.csv"

    # Choose folders with current data 
    amp_folders = ['ampmeter_bottomleft_700GHz_current','ampmeter_bottomright_800GHz_current','ampmeter_middleleft_600GHz_current','ampmeter_middleright_900GHz_current','ampmeter_topRight_1000GHz_current','ampmeter_topleft_500GHz_current']
    amp_base_folder = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\AmpNPower_14_09_21\\Ampmeter14092021sample2021_005_3\\"

    pow_folders = ['bottomleft_700GHz','bottomright_800GHz','middleleft_600GHz','middleright_900GHz','topright_1000GHz']
    pow_base_folder = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\AmpNPower_14_09_21\\Powersupply14092021sample2021_005_3\\"

    # eme_base_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\EmissionEnergy\\500GHz_020721_0\\"
    # eme_tools = AnalysisTools(eme_base_dir)
    # eme_data = eme_tools.get_data(pow_filename, ret=True)
    # eme_positions, eme_voltages, eme_currents = eme_tools.analyse_emission_energy(eme_data)
    # eme_plots = plot_emission_energy_dist(eme_positions[1:], eme_voltages, eme_currents[1:], save_dir=eme_base_dir)

    ### Gets FN fits
    # amp_fn_fits = validity_and_FN_mult(amp_folders, amp_base_folder, amp_filename, plot=True)
    # pow_fn_fits = validity_and_FN_mult(pow_folders, pow_base_folder, pow_filename, yaxis="current3", plot=True)

    ### FN and validity for single file
    # amp_base_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE\\800GHz_020721_current_1\\"
    # validity_and_FN_single(amp_base_dir, amp_filename)

    
    ### Gets emission ceofficients
    # amp_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE\\500GHz_020721_current_0"
    # pow_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent\\500GHz_020721_0"
    # emcoef = get_emcoef(amp_dir, pow_dir, title="EmCoef 0.5THz")

    # amp_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE\\700GHz_020721_current_0"
    # pow_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent\\700GHz_020721_0"
    # emcoef = get_emcoef(amp_dir, pow_dir, title="EmCoef 0.7THz")

    # amp_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE\\800GHz_020721_current_1"
    # pow_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent\\800GHz_020721_0"
    # emcoef = get_emcoef(amp_dir, pow_dir, title="EmCoef 0.8THz")


    # amp_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE\\1000GHz_020721_current"
    # pow_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent\\1000GHz_020721_0"
    # emcoef = get_emcoef(amp_dir, pow_dir, title="EmCoef 1.0THz")



    # # Provide file name
    # amp_filename = "current.csv"
    # pow_filename = "voltage_and_currents.csv"

    # # Choose folders with current data 
    # amp_folders = ['800GHz_020721_current_1','1000GHz_020721_current','700GHz_020721_current_0','500GHz_020721_current_0']
    # amp_base_folder = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE\\"

    # pow_folders = ['800GHz_020721_0','1000GHz_020721_0','700GHz_020721_0','500GHz_020721_0']
    # pow_base_folder = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent\\"

    # eme_base_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\EmissionEnergy\\500GHz_020721_0\\"
    # eme_tools = AnalysisTools(eme_base_dir)
    # eme_data = eme_tools.get_data(pow_filename, ret=True)
    # eme_positions, eme_voltages, eme_currents = eme_tools.analyse_emission_energy(eme_data)
    # eme_plots = plot_emission_energy_dist(eme_positions[1:], eme_voltages, eme_currents[1:], save_dir=eme_base_dir)






