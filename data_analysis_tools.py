# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 10:54:29 2021

@author: willi
"""

import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# import statsmodels.api as sm
import os
pio.renderers.default='svg'

class AnalysisTools:

    def __init__(self, directory=None):
        if directory is not None:
            self.data_directory = directory


    def plot_data(self, data=None, xaxis="time", yaxis="current", position=0, save=False, show=True, model=None, name=None):
        '''Creates an illustration of the measured values in the console'''
# TODO make position into a distinguisher, so other paramters can be chosen based on need
        if data is None:
            try:
                data = self.data
            except:
                raise Exception("Please load and input a dataset")

        if model is not None:
            frame = pd.DataFrame(columns=[xaxis,yaxis,"position", "type"])

            for pos in list(model):
                try:
                    dat = data[data["position"] == pos][xaxis].values.reshape(-1,1)
                except:
                    dat = data[xaxis].values.reshape(-1,1)
                prediction = model[pos].predict(dat)
                f = pd.DataFrame([[d[0],p[0],pos,"fit"] for d,p in zip(dat,prediction)], columns=[xaxis,yaxis,"position","type"])
                frame = frame.append(f)
            
            data["type"] = ["data" for i in range(len(data))]
            data = data.append(frame)

        if position == 'all':
            plot_data = data
            if save:
                fig = px.scatter(plot_data, x=xaxis, y=yaxis, animation_frame="position", color="type")
            else:
                fig = px.scatter(plot_data, x=xaxis, y=yaxis, color='position')

        else:
            if "position" in data.keys():
                plot_data = data.loc[data["position"] == position]
            else:
                plot_data = data
                
            if model is None:
                fig = px.scatter(plot_data, x=xaxis, y=yaxis, title=f"{position}")
            else:
                fit_data = plot_data[plot_data['type'] == 'fit']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_data[plot_data['type'] == 'data'][xaxis], y=plot_data[plot_data['type'] == 'data'][yaxis],name="Data", mode="markers"))
                fig.add_trace(go.Scatter(x=[fit_data[xaxis].iloc[0], fit_data[xaxis].iloc[-1]], y=[fit_data[yaxis].iloc[0], fit_data[yaxis].iloc[-1]], mode="lines", name="Fit"))
                if name:
                    tit = f"{name}"
                else:
                    tit = ""
                fig.update_layout(
                    title = tit,
                    xaxis_title="1/E",
                    yaxis_title="-ln(I/E^2)",
                    # legend_title="Legend Title",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="RebeccaPurple"
                    )
                )

        if show:
            fig.show()
        if save:
            fig.write_html(self.data_directory + "\\" + f"{yaxis}_{position}.html")

    def set_data(self, data):
        self.data = data
            
    def get_data(self, filename, ret=False):
        '''Reads data from saves csv file to pandas'''
        try:
            data = pd.read_csv(os.path.join(self.data_directory + filename))
        except:
            data = pd.read_csv(filename)
            parts = filename.split("\\")
            self.data_directory = ""
            for part in parts[:-1]:
                self.data_directory += part + "\\"
            self.data_directory = self.data_directory[:-1]

        self.data = data
        if ret:
            return data

    def get_background(self, filename, ret=False):
        '''Reads data from saves csv file to pandas and keeps as background variable'''
        try:
            data = pd.read_csv(os.path.join(self.data_directory + filename))
        except:
            data = pd.read_csv(filename)
            parts = filename.split("\\")
            self.data_directory = ""
            for part in parts[:-1]:
                self.data_directory += part + "\\"
            self.data_directory = self.data_directory[:-1]

        self.background = data
        if ret:
            return data

            
    def get_linear_fit(self, x=None, y=None, data=None, xaxis="time", yaxis="current", position=0, plot=False, save_plot=False, show=True, distinguisher="position"):
        '''Returns a linear fit on the input data'''
# TODO make position into a distinguisher, so other paramters can be chosen based on need

        if data is None and x is None and y is None:
            try:
                data = self.data
            except:
                raise Exception("No data is loaded, please insert data")
        if x is None and y is None:
            if position != "all" and distinguisher == "position":
                data = data[data['position'] == position]
                x = data[xaxis].values.reshape(-1,1)
                y = data[yaxis].values.reshape(-1,1)

                model = {position : LinearRegression()}
                model[0].fit(x,y)
                
            else:
                model = {}
                for pos in data[distinguisher].unique():
                    dat = data[data[distinguisher] == pos]
                    x = dat[xaxis].values.reshape(-1,1)
                    y = dat[yaxis].values.reshape(-1,1)
                    
                    model[pos] = LinearRegression()
                    model[pos].fit(x,y)
                    
        else:
            model = {position : LinearRegression()}
            model[position].fit(x,y)

        if plot:
            self.plot_data(data, xaxis=xaxis, yaxis=yaxis, position=position, show=show, save=save_plot, model=model)
        return model


    
    def get_FN_fit(self, data=None, yaxis="current", save=False, plot=True, minimum=0.95, name=None, background=None):
        '''Returns linear fit on the data transformed as (1/E, -ln(I/E^2))'''
        if data is None:
            data = self.data
        
        positions = data["position"].unique()[1:]
        
        if background is None:
            back_name = "background"
            background = data[data["position"] == back_name][yaxis].mean()
            if np.isnan(background):
                back_name = data["position"].unique()[0]
                background = data[data["position"] == back_name][yaxis].mean()
            
        
        # invpowers = np.array([1 / (np.cos(float(pos)*np.pi/180))**2 for pos in positions if pos != back_name])
        invpowers = np.array([1 / (np.cos(float(pos)*np.pi/180)) for pos in positions if pos != back_name]) # TODO use when using cos and not cos^2
        averages = np.array([data[data["position"] == pos][yaxis].mean() - background for pos in data["position"].unique() if pos != back_name])
        
        invpowers = invpowers[averages > 0]
        averages = averages[averages > 0]        
        
        current = [-np.log(av*inv**2) for av,inv in zip(averages,invpowers)]
        
        frame = pd.DataFrame([[p,c,0] for p,c in zip(invpowers,current)], columns=["power","current","position"])
        
        x = frame["power"].values.reshape(-1,1)
        y = frame["current"].values.reshape(-1,1)
        
        model = self.get_linear_fit(x=x, y=y)

        self.validate_linear_model(model, x, y, minimum=minimum)

        # fit_frame = pd.DataFrame([[p,c] for p,c in zip([])])

        if plot:
            self.plot_data(frame,xaxis="power",yaxis="current",model=model, save=save, name=name)
        
        return model
    
    
    def get_mult_FN_fits(self, folders, filename, yaxis="current", save=True, base_folder=None, minimum=0.95, plot=False):
        '''Returns the linear fits for all the FN transformed data found in the folders provided'''
        
        models = {}
        
        for folder in folders:
            
            print(f"Creating model for {folder}")
            
            if base_folder:
                data = self.get_data(base_folder + folder + "\\" + filename, ret=True)
            else:
                data = self.get_data(folder + "\\" + filename, ret=True)
        
            passed = self.validate_data(data=data, yaxis=yaxis, warn=False)
            model = self.get_FN_fit(data, yaxis=yaxis, plot=plot, minimum=minimum, name=folder)
            models[folder] = model[0]

            if not passed:
                print(f"Warning: Data for measure {folder} might be incorrect, validate seperately for more info")
            
            print("\n")

        return models
    
    def validate_data(self, data=None, yaxis="current", warn=True):
        '''Checks the data is somewhat constant during the measurement'''
        
        if data is None:
            data = self.data
        
        positions = data["position"].unique()[1:]
        
        passed = True
        
        for position in positions:
            
            pos_data = data[data["position"] == position]
            
            end_time = pos_data["time"].iloc[-1]
            
            start_avg = pos_data[yaxis][pos_data["time"] < 0.1*end_time].mean()
            end_avg = pos_data[yaxis][pos_data["time"] > 0.9*end_time].mean()

            avg = pos_data[yaxis].mean()
            
            deviation = abs(end_avg - start_avg) / avg * 100
            
            if deviation > 5:
                passed = False

                if warn:
                    print(f"\nWARNING: Position {position} has deviation {deviation}% \n")
            
        return passed

    def validate_linear_model(self, models, x, y, minimum=0.95):
        
        for model in models:
            score = models[model].score(x, y)
            
            if score < minimum:
                print(f"Warning: Linear model {model} has score {score} (lower than {minimum})")
            else:
                print(f"Model score {score}")
        return score
    
    # def get_coef_uncertainty(self, x, y):
    #     ols = sm.OLS(y,x)
    #     return ols.fit()
        # ols_result.HCO_se
        # ols_result.cov_HC0
        
    def plot_current_as_E(self, data, yaxis="current"):

        positions = data["position"].unique()[1:]
        # Es = (np.cos(positions * np.pi / 180))**2
        Es = (np.cos(positions * np.pi / 180))
        currs = np.array([data[data["position"] == pos][yaxis].mean() for pos in positions])

        background = data["position"].unique()[0]
        back_curr = data[data["position"] == background][yaxis].mean()

        plot_data = pd.DataFrame([[e,c-back_curr,le,lc] for e,c,le,lc in zip(Es, currs, np.log(Es), np.log(currs -back_curr))], columns=["E-field", yaxis, "ln(E-field)", f"ln({yaxis})"])

        fig = px.scatter(plot_data, x="E-field", y=yaxis)
        fig.show()
        fig = px.scatter(plot_data, x="E-field", y=f"ln({yaxis})")
        fig.show()
        fig = px.scatter(plot_data, x="ln(E-field)", y=f"ln({yaxis})")        
        fig.show()

        return plot_data

    def analyse_emission_energy(self, data=None):
        
        if data is None:
            data = self.data
            
        positions = data.position.unique()
        voltages = data.voltage_setting.unique()
        
        currents = np.empty((len(positions),len(voltages)))

        for i,pos in enumerate(positions):
            for j,vol in enumerate(voltages):
                currents[i,j] = data[(data["position"]==pos) & (data["voltage_setting"]==vol)]["current3"].mean()

        return positions, voltages, currents




if __name__ == '__main__':
    
    #TODO from now on only use "background" = 90. Det andet er kaos i typper i pandas arrays...
    
    base_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\IasE\\500GHz_020721_current_0"
    filename = "\\current.csv"
    
    # base_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent\\1000GHz_020721_0"
    # filename = "\\voltage_and_currents.csv"

    tools = AnalysisTools()
    data = tools.get_data(base_dir + filename, ret=True)
    # tt = tools.plot_current_as_E(data, yaxis="current")
    # mod = tools.get_linear_fit(data=data, plot=True, position="all", save_plot=True)
    # fn_fit = tools.get_FN_fit(data)

    # base_dir = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\PhosphorCurrent\\1000GHz_020721_0"
    # filename = "\\voltage_and_currents.csv"
    
    # tools = AnalysisTools()
    # data = tools.get_data(base_dir + filename, ret=True)
    # mod = tools.get_linear_fit(data=data, plot=False, position="all", save_plot=True, yaxis="current3")
    # fn_fit = tools.get_FN_fit(data, yaxis="current3")