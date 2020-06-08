#!/usr/bin/env python3

'''
COMPUTING PROJECT
PROJECT C: PHASE TRANSITIONS IN THE ISING AND XY MODELS

Two-dimensional Ising and XY models based on the Metropolis algorithm, written in Python with a TkInter user interface
Helical boundary conditions used such that the spin lattice can be represented as a single 1D array

MINIMUM PROPERTIES INVESTIGATED
(1) Thermalisation
(2) Autocovariance and Autocorrelation
(3) Mean Magnetisation
(4) Heat Capacity
(5) Finite-Size Scaling
(6) Hysteresis

ADDITIONAL PROPERTIES INVESTIGATED
(7) Correlation Length
(8) Magnetic Susceptibility
(9) Spin Stiffness (Helicity Modulus)

*** ADDITIONAL MODULES REQUIRED ***
NumPy, SciPy
TkInter            for interface
PIL (Pillow)       for spin lattice display
Matplotlib
Time
Cmath
'''

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import time
import cmath
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from scipy.optimize import curve_fit

def exit():
    ''' Quits program '''
    root.quit()

# TTK Application
class MainApplication(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        # Colour maps from matplotlib
        self.colourmap_binary = plt.get_cmap('binary') # For Ising model
        self.colourmap = plt.get_cmap('twilight') # Use a cyclic colour map such that 0+ and 2*pi- radians are equivalently displayed in XY model
        
        '''
        —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        ——————————————————————————————————————————————————————     HOLDERS     ——————————————————————————————————————————————————————
        —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        '''
        
        # Conditions
        self.img_handler = False                                      # = True if simulation is to run
        self.update_en = False                                        # Continuously updates energy
        self.update_mag = False                                       # Continuously updates magnetisation
        
        # Holder for spin data (in helical coordinates)
        self.data = []

        # Holders for measurement data
        self.ave_energy = []
        self.ave_mag = []
        self.mag = []
        self.A = []
        self.mag2 = []
        self.corl = []
        self.E = []
        self.M = []
        
        # Model --- see option menu
        self.ModelList = [
        "Ising Model",
        "XY Model"
        ]
        
        # Initial Conditions --- see option menu
        self.InitialList = [
        "Spin-Aligned",
        "Randomised"
        ]
        
        # Evolution --- see option menu
        self.EvolutionList = [
        "Alternating",
        "Simultaneous",
        "Systematic",
        "Randomised"
        ]
        
        # Option Menu variables
        self.initial = tk.StringVar(self)                             # Choice of initial condition
        self.initial.set(self.InitialList[0])
        self.evolution = tk.StringVar(self)                           # Choice of evolution method
        self.evolution.set(self.EvolutionList[0])
        self.model = tk.StringVar(self)                               # Choice of model
        self.model.set(self.ModelList[0])
        
        # Holders for main variables
        self.run = False                                              # Condition for simulation to start or resume running
        self.temp = tk.DoubleVar(); self.temp.set(2)                  # Temperature of system
        self.H = tk.DoubleVar(); self.H.set(0)                        # Imposed H-field strength
        self.J = tk.DoubleVar(); self.J.set(1)                        # Exchange energy
        self.N = tk.IntVar(); self.N.set(0)                           # Lattice size N
        self.Hth = tk.DoubleVar(); self.Hth.set(0)                    # Imposed H-field angle (for XY model)
        self.stepcount = tk.IntVar(); self.stepcount.set(0)           # Number of steps already taken
        self.contupdate = tk.IntVar(); self.contupdate.set(0)         # Checkbutton variable for continuous update
        self.runtime = tk.IntVar(); self.runtime.set(1)               # Checkbutton variable for showing run time in command prompt
        
        # Task (1) variables --- thermalisation
        self.thermdecay = tk.IntVar(); self.thermdecay.set(0)         # Checkbutton variable for decay plot
        self.temprange4 = tk.DoubleVar(); self.temprange4.set(1.5)    # Lower temperature bound
        self.temprange5 = tk.DoubleVar(); self.temprange5.set(3)      # Upper temperature bound
        self.samp3 = tk.IntVar(); self.samp3.set(20)                  # Number of temperature sampling points
        self.thermstep = tk.IntVar(); self.thermstep.set(100)         # Number of steps for each temperature
        self.thermflips = tk.IntVar(); self.thermflips.set(1)         # Number of times energy crosses threshold to be considered 'thermalised'
        
        # Task (2) variables --- autocorrelation
        self.tlagbuffer = tk.IntVar(); self.tlagbuffer.set(10)        # Time lag buffer
        self.temprange6 = tk.DoubleVar(); self.temprange6.set(1.5)    # Lower temperature bound
        self.temprange7 = tk.DoubleVar(); self.temprange7.set(3)      # Upper temperature bound
        self.samp4 = tk.IntVar(); self.samp4.set(20)                  # Number of temperature sampling points
        self.corstep = tk.IntVar(); self.corstep.set(100)             # Number of measurement steps for each temperature
        
        # Task (3) variables --- magnetisation
        self.temprange0 = tk.DoubleVar(); self.temprange0.set(0)      # Lower temperature bound
        self.temprange = tk.DoubleVar(); self.temprange.set(3)        # Upper temperature bound
        self.samp = tk.IntVar(); self.samp.set(30)                    # Number of temperature sampling points
        self.tbuffer = tk.IntVar(); self.tbuffer.set(10)              # Thermalisation buffer time
        self.magstep = tk.IntVar(); self.magstep.set(50)              # Number of measurement steps for each temperature
        
        # Task (4) variables --- heat capacity
        self.temprange2 = tk.DoubleVar(); self.temprange2.set(1.5)    # Lower temperature bound
        self.temprange3 = tk.DoubleVar(); self.temprange3.set(3)      # Upper temperature bound
        self.samp2 = tk.IntVar(); self.samp2.set(20)                  # Number of temperature sampling points
        self.tbuffer2 = tk.IntVar(); self.tbuffer2.set(10)            # Thermalisation buffer time
        self.capstep = tk.IntVar(); self.capstep.set(100)             # Number of energy (heat capacity) measurement steps for each temperature
        
        # Task (5) variables --- finite-size scaling
        self.Ndown = tk.IntVar(); self.Ndown.set(5)                   # Lower lattice size bound
        self.Nup = tk.IntVar(); self.Nup.set(101)                     # Upper lattice size bound
        self.Nnum = tk.IntVar(); self.Nnum.set(20)                    # Number of lattice sizes considered
        
        # Task (6) variables --- hysteresis
        self.Hperiod = tk.IntVar(); self.Hperiod.set(100)             # Period of oscillation of external H (in time steps)
        self.Hamp = tk.DoubleVar(); self.Hamp.set(1)                  # Amplitude of oscillation of external H
        
        # Task (7) variables --- correlation length
        self.decay7 = tk.IntVar(); self.decay7.set(0)                 # Checkbutton variable for decay plot
        self.temprange8 = tk.DoubleVar(); self.temprange8.set(1.5)    # Lower temperature bound
        self.temprange9 = tk.DoubleVar(); self.temprange9.set(3)      # Upper temperature bound
        self.samp5 = tk.IntVar(); self.samp5.set(20)                  # Number of temperature sampling points
        self.tbuffer3 = tk.IntVar(); self.tbuffer3.set(10)            # Thermalisation buffer time
        self.corlenstep = tk.IntVar(); self.corlenstep.set(20)        # Number of measurement steps for each temperature
        self.rep = tk.IntVar(); self.rep.set(10)                      # Number of times simulation is repeated
        
        # Task (8) variables --- magnetic susceptibility
        self.temprange10 = tk.DoubleVar(); self.temprange10.set(1.5)  # Lower temperature bound
        self.temprange11 = tk.DoubleVar(); self.temprange11.set(3)    # Upper temperature bound
        self.samp6 = tk.IntVar(); self.samp6.set(20)                  # Number of temperature sampling points
        self.tbuffer4 = tk.IntVar(); self.tbuffer4.set(10)            # Thermalisation buffer time
        self.susstep = tk.IntVar(); self.susstep.set(100)             # Number of measurement steps for each temperature
        
        # Task (9) variables --- spin stiffness
        self.temprange12 = tk.DoubleVar(); self.temprange12.set(0.8)  # Lower temperature bound
        self.temprange13 = tk.DoubleVar(); self.temprange13.set(1.2)  # Upper temperature bound
        self.samp7 = tk.IntVar(); self.samp7.set(20)                  # Number of temperature sampling points
        self.tbuffer5 = tk.IntVar(); self.tbuffer5.set(1000)          # Thermalisation buffer time
        self.spinstep = tk.IntVar(); self.spinstep.set(1000)          # Number of measurement steps for each temperature
        self.spinrep = tk.IntVar(); self.spinrep.set(10)              # NUmber of repetitions
        
        '''
        ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        ——————————————————————————————————————————————————————     INTERFACE LAYOUT     ——————————————————————————————————————————————————————
        ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        '''
        
        # Frames
        self.gridview = ttk.LabelFrame(self, text = "———————————————   SPIN LATTICE   ———————————————")
        self.controls = ttk.LabelFrame(self, text = "—————————————————————————   CONTROLS   —————————————————————————")
        self.measurements = ttk.LabelFrame(self, text = "———   MEASUREMENTS   ———")
        
        # Controls
        self.controls.temperature_label = ttk.Label(self.controls, text = "T", width = 3)
        self.controls.temperature_entry = ttk.Entry(self.controls, textvariable = self.temp, width = 5)
        self.controls.H_label = ttk.Label(self.controls, text = "H", width = 3)
        self.controls.H_entry = ttk.Entry(self.controls, textvariable = self.H, width = 5)
        self.controls.N_button = ttk.Button(self.controls, text = "Set N", command = self.set_N)
        self.controls.N_entry = ttk.Entry(self.controls, textvariable = self.N, width = 5)
        self.controls.initial_label = ttk.Label(self.controls, text = "Initial Conditions")
        self.controls.initial_menu = ttk.OptionMenu(self.controls, self.initial, self.InitialList[0], *self.InitialList)
        self.controls.evolution_label = ttk.Label(self.controls, text = "Evolution")
        self.controls.evolution_menu = ttk.OptionMenu(self.controls, self.evolution, self.EvolutionList[0], *self.EvolutionList)
        self.controls.model_label = ttk.Label(self.controls, text = "Model")
        self.controls.model_menu = ttk.OptionMenu(self.controls, self.model, self.ModelList[0], *self.ModelList)
        self.controls.J_label = ttk.Label(self.controls, text = "J", width = 3)
        self.controls.J_entry = ttk.Entry(self.controls, textvariable = self.J, width = 5)
        self.controls.Hth_label = ttk.Label(self.controls, text = "H (angle)")
        self.controls.Hth_entry = ttk.Entry(self.controls, textvariable = self.Hth, width = 5)
        
        self.controls.temperature_label.grid(column = 6, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.temperature_entry.grid(column = 7, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.H_label.grid(column = 4, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.H_entry.grid(column = 5, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.N_button.grid(column = 2, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.N_entry.grid(column = 3, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.J_label.grid(column = 4, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.J_entry.grid(column = 5, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.Hth_label.grid(column = 6, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.Hth_entry.grid(column = 7, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.initial_label.grid(column = 2, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.initial_menu.grid(column = 3, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.initial_menu.configure(width = 15)
        self.controls.evolution_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.evolution_menu.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.evolution_menu.configure(width = 15)
        self.controls.model_label.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.model_menu.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.controls.model_menu.configure(width = 15)
        
        # Gridview
        self.canvas1 = tk.Canvas(self.gridview, width = 500, height = 500)
        self.canvas1.pack(side = "bottom", fill = "both", expand = "yes")
        self.gridview.start = ttk.Button(self.gridview, text = "Start", command = self.start_simulation)
        self.gridview.stop = ttk.Button(self.gridview, text = "Stop", command = self.stop_simulation)
        self.gridview.set = ttk.Button(self.gridview, text = "Set/Reset Lattice", command = self.set_lattice)
        self.gridview.quit = ttk.Button(self.gridview, text = "Exit", command = exit)
        self.gridview.stepcount_label = ttk.Label(self.gridview, text = "Step count:")
        self.gridview.stepcount_output = ttk.Label(self.gridview, textvariable = self.stepcount)
        self.gridview.resetstepcount = ttk.Button(self.gridview, text = "Reset step count", command = self.reset_step)
        self.gridview.update_checklabel = ttk.Label(self.gridview, text = "Continuous image update")
        self.gridview.update_checkbox = ttk.Checkbutton(self.gridview, variable = self.contupdate)
        
        self.canvas1.grid(column = 0, row = 4, columnspan = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.gridview.start.grid(column = 0, row = 0, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        self.gridview.stop.grid(column = 1, row = 0, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        self.gridview.set.grid(column = 2, row = 0, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        self.gridview.quit.grid(column = 3, row = 0, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        self.gridview.stepcount_label.grid(column = 0, row = 2, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        self.gridview.stepcount_output.grid(column = 1, row = 2, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        self.gridview.resetstepcount.grid(column = 2, row = 2, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        self.gridview.update_checklabel.grid(column = 0, row = 3, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        self.gridview.update_checkbox.grid(column = 1, row = 3, sticky = (tk.E, tk.W), padx = 3, pady = 3)
        
        # Measurements
        self.measurements.runtime_checklabel = ttk.Label(self.measurements, text = "Show run time in prompt")
        self.measurements.runtime_checkbox = ttk.Checkbutton(self.measurements, variable = self.runtime)
        self.measurements.thermalise_button = ttk.Button(self.measurements, text = "(1) Thermalise", command = self.therm)
        self.measurements.autocorrelation_button = ttk.Button(self.measurements, text = "(2) Autocorrelation", command = self.autocor)
        self.measurements.meanmagnetisation_button = ttk.Button(self.measurements, text = "(3) Mean Magnetisation", command = self.meanmag)
        self.measurements.heatcapacity_button = ttk.Button(self.measurements, text = "(4) Heat Capacity", command = self.heatcap)
        self.measurements.scaling_button = ttk.Button(self.measurements, text = "(5) Finite-Size Scaling", command = self.fs_scaling)
        self.measurements.hysteresis_button = ttk.Button(self.measurements, text = "(6) Hysteresis", command = self.hys)
        self.measurements.correlation_button = ttk.Button(self.measurements, text = "(7) Correlation Length", command = self.corlen)
        self.measurements.susceptibility_button = ttk.Button(self.measurements, text = "(8) Magnetic Susceptibility", command = self.sus)
        self.measurements.spin_button = ttk.Button(self.measurements, text = "(9) Spin Stiffness", command = self.spin)
        
        self.measurements.runtime_checklabel.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.runtime_checkbox.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.thermalise_button.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.autocorrelation_button.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.meanmagnetisation_button.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.heatcapacity_button.grid(column = 0, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.scaling_button.grid(column = 0, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.hysteresis_button.grid(column = 0, row = 6, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.correlation_button.grid(column = 0, row = 7, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.susceptibility_button.grid(column = 0, row = 8, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        self.measurements.spin_button.grid(column = 0, row = 9, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        
        # Assemble main frames
        self.gridview.grid(column = 0, row = 1, columnspan = 2, sticky = (tk.N, tk.S, tk.W), padx = 3, pady = 3, ipadx = 3, ipady = 12)
        self.controls.grid(column = 0, row = 0, columnspan = 3, sticky = (tk.N, tk.S, tk.W), padx = 3, pady = 3, ipadx = 3, ipady = 12)
        self.measurements.grid(column = 2, row = 1, rowspan = 2, columnspan = 1, sticky = (tk.N, tk.W, tk.E), padx = 3, pady = 3, ipadx = 3, ipady = 12)
    
    '''
    ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     MAIN FUNCTIONS     ————————————————————————————————————————————————————
    ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def set_N(self, *args):
        ''' Resets lattice to a given size N '''
        N = np.int(self.N.get())
        N = np.absolute(N)
        model = self.model.get()
        self.N.set(N)
        
        if model == "Ising Model":
            self.data = np.ones((N**2), dtype = int)
        else:
            # XY Model
            array = np.ones(N**2)*2*np.pi
            self.data = np.array(array, dtype = float)
        
        self.set_lattice()
    
    def set_lattice(self, *args):
        ''' Sets/resets lattice before starting a simulation '''
        select = self.initial.get()
        model = self.model.get()
        
        if select == "Randomised":
            N = self.N.get()
            
            if model == "Ising Model":
                array = np.random.randint(2, size = N**2)
                array = 2*(array - 0.5)
                self.data = np.array(array, dtype = int)
            else:
                array = np.random.random_sample((N**2,))
                array = 2*np.pi*array
                self.data = np.array(array, dtype = float)
            
        elif select == "Spin-Aligned":
            N = self.N.get()
            
            if model == "Ising Model":
                self.data = np.ones((N**2), dtype = int)
            else:
                array = np.ones(N**2)*2*np.pi
                self.data = np.array(array, dtype = float)
            
        else:
            messagebox.showinfo("Error", "Error setting lattice")
        
        self.update()
    
    def update(self, *args):
        ''' Functions to be called when the canvas frame is updated '''
        self.update_canvas()
        # Add other functions here if necessary
    
    def update_canvas(self, *args):
        ''' Updates representation of spins on canvas '''
        array = self.data
        N = self.N.get()
        model = self.model.get()
        array2 = (np.reshape(array, (N, N))) # convert linear N**2 spin array to an N x N matrix
        
        if model == "Ising Model":
            array3 = (-(array2/2) + 0.5) # convert [-1, 1] representation to [1, 0]
            col_im = self.colourmap_binary(array3)
            img = Image.fromarray((col_im[:,:,:3]*126 + 124).astype(np.uint8)) # Convert data to image (8-bit grayscale)
        else:
            array3 = array2/(2*np.pi)
            col_im = self.colourmap(array3)
            img = Image.fromarray((col_im[:,:,:3]*255).astype(np.uint8)) # Convert data to image (colourised)
        
        can_w = np.int(self.canvas1.cget('width'))
        can_h = np.int(self.canvas1.cget('height'))
        img2 = img.resize((can_w, can_h), Image.NEAREST) # Resize image to fit canvas
        self.img3 = ImageTk.PhotoImage(img2)
        
        if self.img_handler:
            self.canvas1.delete(self.img_handler) # Remove previous image
        self.img_handler = self.canvas1.create_image(can_w/2, can_h/2, image = self.img3) # Update image
    
    def single_step(self, i):
        ''' Decides whether to flip/rotate i-th spin site or leave it alone. Returns change in spin sign if Ising Model, and new spin if XY Model '''
        data = np.array(self.data)
        model = self.model.get()
        s = data[i] # i-th element of lattice (helical B.C.s)
        N = self.N.get()
        J = self.J.get()
        H = self.H.get()
        
        # Four adjacent elements of i-th spin site
        i_left = (i - 1)%(N**2)
        s_left = data[i_left]
        i_right = (i + 1)%(N**2)
        s_right = data[i_right]
        i_up = (i - N)%(N**2)
        s_up = data[i_up]
        i_down = (i + N)%(N**2)
        s_down = data[i_down]
        
        if model == "Ising Model":
            # s2 = -1 if spin is to be flipped and s2 = 1 if left alone
            dE = 2*s*(J*(s_left + s_right + s_up + s_down) + H) # Energy required to flip spin, noting that minus signs cancel
            s2 = 1 # Default: no flip
            
            if dE < 0:
                s2 = -1
            else:
                T = self.temp.get()
                if T == 0:
                    prob = 0
                else:
                    prob = np.exp(-dE/T)
                    p = np.random.random_sample()
                    if prob > p:
                        s2 = -1
                    else:
                        s2 = 1
        else:
            # s2 is proposed new spin
            sn = 2*np.pi*np.random.random_sample() # Propose new spin angle
            Hth = self.Hth.get()
            Eold = -J*(np.cos(s - s_left) + np.cos(s - s_right) + np.cos(s - s_up) + np.cos(s - s_down)) - H*np.cos(s - Hth)
            Enew = -J*(np.cos(sn - s_left) + np.cos(sn - s_right) + np.cos(sn - s_up) + np.cos(sn - s_down)) - H*np.cos(sn - Hth)
            dE = Enew - Eold # Energy required to rotate spin
            s2 = s # Default: no rotation
            
            if dE < 0:
                s2 = sn
            else:
                T = self.temp.get()
                if T == 0:
                    prob = 0
                else:
                    prob = np.exp(-dE/T)
                    p = np.random.random_sample()
                    if prob > p:
                        s2 = sn
                    else:
                        s2 = s
        return s2
    
    def flip_one(self, dE):
        ''' Given an energy dE required to flip a single spin, decides whether to flip it or leave it alone. Vectorised under alt/sim_step_metro '''
        # s2 = -1 if spin is to be flipped and s2 = 1 if left alone
        s2 = 1 # Default: no flip
        
        if dE < 0:
            s2 = -1
        else:
            T = self.temp.get()
            if T == 0:
                prob = 0
            else:
                prob = np.exp(-dE/T)
                p = np.random.random_sample()
                if prob > p:
                    s2 = -1
                else:
                    s2 = 1
        return s2
    
    def rotate_one(self, dE, data, s_new):
        ''' Given an energy dE required to rotate a single spin, decides whether to rotate it or leave it alone. Vectorised under alt/sim_step_metro '''
        # s2 = s_new if spin is to be rotated and s2 = data if left alone
        s2 = data # Default: no rotation
        
        if dE < 0:
            s2 = s_new
        else:
            T = self.temp.get()
            if T == 0:
                prob = 0
            else:
                prob = np.exp(-dE/T)
                p = np.random.random_sample()
                if prob > p:
                    s2 = s_new
                else:
                    s2 = data
        return s2
    
    def alt_step_metro(self, *args):
        ''' Evolves all spin sites a single step forward in two steps, in an alternating "checkerboard" pattern --- valid only for odd N '''
        N = self.N.get()
        H = self.H.get()
        model = self.model.get()
        Nsq = N**2
        
        # Require odd N for 'checkerboard' update
        if Nsq % 2 == 0:
            messagebox.showinfo("Notification", "Alternating evolution valid only for odd N")
            self.run = False
        elif Nsq % 2 == 1:
            if self.run == True:
                if self.contupdate.get(): # Updates lattice after each alternating step
                    J = self.J.get()
                    
                    # Deal with even-numbered spins
                    data = np.array(self.data)
                    data_even = data[::2]                     # Pick out even-numbered elements
                    data_odd = data[1::2]                     # Pick out odd-numbered elements
                    
                    s_left1 = data[-1]
                    s_left2 = data[:-1]
                    s_left = np.append(s_left1, s_left2)      # Array containing spin states of elements directly left of those in original data array
                    s_left_even = s_left[::2]
                    
                    s_right1 = data[1:]
                    s_right2 = data[0]
                    s_right = np.append(s_right1, s_right2)   # Directly right
                    s_right_even = s_right[::2]
                    
                    s_up1 = data[-N:]
                    s_up2 = data[:-N]
                    s_up = np.append(s_up1, s_up2)            # Directly above
                    s_up_even = s_up[::2]
                    
                    s_down1 = data[N:]
                    s_down2 = data[:N]
                    s_down = np.append(s_down1, s_down2)      # Directly below
                    s_down_even = s_down[::2]
                    
                    N_even = np.int(Nsq/2 + 0.5)              # Number of even elements
                    
                    if model == "Ising Model":
                        gesamt_even = np.array(J*(s_left_even + s_right_even + s_up_even + s_down_even) + H*np.ones(N_even))
                        dE_even = 2*np.multiply(data_even, gesamt_even) # Energy required to flip each even-numbered spin, as an array
                        vec = np.vectorize(self.flip_one)
                        flip_even = vec(dE_even)
                        data2_even = np.multiply(data_even, flip_even)
                        
                        # Join new even-numbered array with unmodified odd-numbered array
                        data2 = np.empty((data2_even.size + data_odd.size,), dtype = data2_even.dtype)
                        data2[::2] = data2_even
                        data2[1::2] = data_odd
                    else:
                        Hth = self.Hth.get()
                        Hth_array_even = np.ones(np.int(Nsq/2 + 0.5))*Hth
                        E0_even = np.array(-J*(np.cos(data_even - s_left_even) + np.cos(data_even - s_right_even) + np.cos(data_even - s_up_even) + np.cos(data_even - s_down_even)) - H*np.cos(data_even - Hth_array_even))
                        s_new_even = np.random.random_sample(np.int(Nsq/2 + 0.5),)*2*np.pi # Proposed new spin array
                        E1_even = np.array(-J*(np.cos(s_new_even - s_left_even) + np.cos(s_new_even - s_right_even) + np.cos(s_new_even - s_up_even) + np.cos(s_new_even - s_down_even)) - H*np.cos(s_new_even - Hth_array_even))
                        dE_even = E1_even - E0_even # Energy required to rotate each even-numbered spin
                        vec = np.vectorize(self.rotate_one)
                        data2_even = vec(dE_even, data_even, s_new_even)
                        
                        # Join new even-numbered array with unmodified odd-numbered array
                        data2 = np.empty((data2_even.size + data_odd.size,), dtype = data2_even.dtype)
                        data2[::2] = data2_even
                        data2[1::2] = data_odd
                    
                    self.data = data2
                    self.update()
                    root.update_idletasks()
                    root.update()
                    
                    # Deal with odd-numbered spins in an analogous manner
                    data = np.array(self.data)
                    data_even = data[::2]
                    data_odd = data[1::2]
                    
                    s_left1 = data[-1]
                    s_left2 = data[:-1]
                    s_left = np.append(s_left1, s_left2)
                    s_left_odd = s_left[1::2]
                    
                    s_right1 = data[1:]
                    s_right2 = data[0]
                    s_right = np.append(s_right1, s_right2)
                    s_right_odd = s_right[1::2]
                    
                    s_up1 = data[-N:]
                    s_up2 = data[:-N]
                    s_up = np.append(s_up1, s_up2)
                    s_up_odd = s_up[1::2]
                    
                    s_down1 = data[N:]
                    s_down2 = data[:N]
                    s_down = np.append(s_down1, s_down2)
                    s_down_odd = s_down[1::2]
                    
                    N_odd = np.int(Nsq/2 - 0.5)
                    
                    if model == "Ising Model":
                        gesamt_odd = np.array(J*(s_left_odd + s_right_odd + s_up_odd + s_down_odd) + H*np.ones(N_odd))
                        dE_odd = 2*np.multiply(data_odd, gesamt_odd)
                        flip_odd = vec(dE_odd)
                        data3_odd = np.multiply(data_odd, flip_odd)
                        
                        data3 = np.empty((data_even.size + data3_odd.size,), dtype = data_even.dtype)
                        data3[::2] = data_even
                        data3[1::2] = data3_odd
                    else:
                        Hth = self.Hth.get()
                        Hth_array_odd = np.ones(np.int(Nsq/2 - 0.5))*Hth
                        E0_odd = np.array(-J*(np.cos(data_odd - s_left_odd) + np.cos(data_odd - s_right_odd) + np.cos(data_odd - s_up_odd) + np.cos(data_odd - s_down_odd)) - H*np.cos(data_odd - Hth_array_odd))
                        s_new_odd = np.random.random_sample(np.int(Nsq/2 - 0.5),)*2*np.pi
                        E1_odd = np.array(-J*(np.cos(s_new_odd - s_left_odd) + np.cos(s_new_odd - s_right_odd) + np.cos(s_new_odd - s_up_odd) + np.cos(s_new_odd - s_down_odd)) - H*np.cos(s_new_odd - Hth_array_odd))
                        dE_odd = E1_odd - E0_odd
                        vec = np.vectorize(self.rotate_one)
                        data3_odd = vec(dE_odd, data_odd, s_new_odd)
                        
                        data3 = np.empty((data_even.size + data3_odd.size,), dtype = data_even.dtype)
                        data3[::2] = data_even
                        data3[1::2] = data3_odd
                    
                    self.data = data3
                    self.update()
                    root.update_idletasks()
                    root.update()
                    
                    count = self.stepcount.get()
                    count += 1
                    self.stepcount.set(count)
                
                else: # Updates lattice only after both alternating steps have been done
                    J = self.J.get()
                    
                    # Deal with even-numbered spins in an analogous manner
                    data = np.array(self.data)
                    data_even = data[::2]
                    data_odd = data[1::2]
                    
                    s_left1 = data[-1]
                    s_left2 = data[:-1]
                    s_left = np.append(s_left1, s_left2)
                    s_left_even = s_left[::2]
                    
                    s_right1 = data[1:]
                    s_right2 = data[0]
                    s_right = np.append(s_right1, s_right2)
                    s_right_even = s_right[::2]
                    
                    s_up1 = data[-N:]
                    s_up2 = data[:-N]
                    s_up = np.append(s_up1, s_up2)
                    s_up_even = s_up[::2]
                    
                    s_down1 = data[N:]
                    s_down2 = data[:N]
                    s_down = np.append(s_down1, s_down2)
                    s_down_even = s_down[::2]
                    
                    N_even = np.int(Nsq/2 + 0.5)
                    
                    if model == "Ising Model":
                        gesamt_even = np.array(J*(s_left_even + s_right_even + s_up_even + s_down_even) + H*np.ones(N_even))
                        dE_even = 2*np.multiply(data_even, gesamt_even)
                        vec = np.vectorize(self.flip_one)
                        flip_even = vec(dE_even)
                        data2_even = np.multiply(data_even, flip_even)
                        
                        data2 = np.empty((data2_even.size + data_odd.size,), dtype = data2_even.dtype)
                        data2[::2] = data2_even
                        data2[1::2] = data_odd
                    else:
                        Hth = self.Hth.get()
                        Hth_array_even = np.ones(np.int(Nsq/2 + 0.5))*Hth
                        E0_even = np.array(-J*(np.cos(data_even - s_left_even) + np.cos(data_even - s_right_even) + np.cos(data_even - s_up_even) + np.cos(data_even - s_down_even)) - H*np.cos(data_even - Hth_array_even))
                        s_new_even = np.random.random_sample(np.int(Nsq/2 + 0.5),)*2*np.pi
                        E1_even = np.array(-J*(np.cos(s_new_even - s_left_even) + np.cos(s_new_even - s_right_even) + np.cos(s_new_even - s_up_even) + np.cos(s_new_even - s_down_even)) - H*np.cos(s_new_even - Hth_array_even))
                        dE_even = E1_even - E0_even
                        vec = np.vectorize(self.rotate_one)
                        data2_even = vec(dE_even, data_even, s_new_even)
                        
                        data2 = np.empty((data2_even.size + data_odd.size,), dtype = data2_even.dtype)
                        data2[::2] = data2_even
                        data2[1::2] = data_odd
                    
                    self.data = data2
                    
                    # Deal with odd-numbered spins in an analogous manner
                    data = np.array(self.data)
                    data_even = data[::2]
                    data_odd = data[1::2]
                    
                    s_left1 = data[-1]
                    s_left2 = data[:-1]
                    s_left = np.append(s_left1, s_left2)
                    s_left_odd = s_left[1::2]
                    
                    s_right1 = data[1:]
                    s_right2 = data[0]
                    s_right = np.append(s_right1, s_right2)
                    s_right_odd = s_right[1::2]
                    
                    s_up1 = data[-N:]
                    s_up2 = data[:-N]
                    s_up = np.append(s_up1, s_up2)
                    s_up_odd = s_up[1::2]
                    
                    s_down1 = data[N:]
                    s_down2 = data[:N]
                    s_down = np.append(s_down1, s_down2)
                    s_down_odd = s_down[1::2]
                    
                    N_odd = np.int(Nsq/2 - 0.5)
                    
                    if model == "Ising Model":
                        gesamt_odd = np.array(J*(s_left_odd + s_right_odd + s_up_odd + s_down_odd) + H*np.ones(N_odd))
                        dE_odd = 2*np.multiply(data_odd, gesamt_odd)
                        flip_odd = vec(dE_odd)
                        data3_odd = np.multiply(data_odd, flip_odd)
                        
                        data3 = np.empty((data_even.size + data3_odd.size,), dtype = data_even.dtype)
                        data3[::2] = data_even
                        data3[1::2] = data3_odd
                    else:
                        Hth = self.Hth.get()
                        Hth_array_odd = np.ones(np.int(Nsq/2 - 0.5))*Hth
                        E0_odd = np.array(-J*(np.cos(data_odd - s_left_odd) + np.cos(data_odd - s_right_odd) + np.cos(data_odd - s_up_odd) + np.cos(data_odd - s_down_odd)) - H*np.cos(data_odd - Hth_array_odd))
                        s_new_odd = np.random.random_sample(np.int(Nsq/2 - 0.5),)*2*np.pi
                        E1_odd = np.array(-J*(np.cos(s_new_odd - s_left_odd) + np.cos(s_new_odd - s_right_odd) + np.cos(s_new_odd - s_up_odd) + np.cos(s_new_odd - s_down_odd)) - H*np.cos(s_new_odd - Hth_array_odd))
                        dE_odd = E1_odd - E0_odd
                        vec = np.vectorize(self.rotate_one)
                        data3_odd = vec(dE_odd, data_odd, s_new_odd)
                        
                        data3 = np.empty((data_even.size + data3_odd.size,), dtype = data_even.dtype)
                        data3[::2] = data_even
                        data3[1::2] = data3_odd
                    
                    self.data = data3
                    self.update()
                    root.update_idletasks()
                    root.update()
                    
                    count = self.stepcount.get()
                    count += 1
                    self.stepcount.set(count)
            else:
                return
        else:
            messagebox.showinfo("Error", "Invalid N")
    
    def sim_step_metro(self, *args):
        ''' Simultaneously evolves all spin sites a single step forward '''
        model = self.model.get()
        
        if self.run == True:
            N = self.N.get()
            J = self.J.get()
            H = self.H.get()
            data = np.array(self.data)
            
            # Re-arrange array such that indices refer to spins directly left/right/up/down
            s_left1 = data[-1]
            s_left2 = data[:-1]
            s_left = np.append(s_left1, s_left2)
            
            s_right1 = data[1:]
            s_right2 = data[0]
            s_right = np.append(s_right1, s_right2)
            
            s_up1 = data[-N:]
            s_up2 = data[:-N]
            s_up = np.append(s_up1, s_up2)
            
            s_down1 = data[N:]
            s_down2 = data[:N]
            s_down = np.append(s_down1, s_down2)
            
            if model == "Ising Model":
                gesamt = np.array(J*(s_left + s_right + s_up + s_down) + H*np.ones(N**2)) # Energy of each spin as an array
                dE = 2*np.multiply(data, gesamt) # Change in energy if each single spin were flipped and the rest kept the same, as an array
                vec = np.vectorize(self.flip_one)
                flip = vec(dE)
                data2 = np.multiply(data, flip) # New spin state array
            else:
                Hth = self.Hth.get()
                Hth_array = np.ones(N**2)*Hth
                E0 = np.array(-J*(np.cos(data - s_left) + np.cos(data - s_right) + np.cos(data - s_up) + np.cos(data - s_down)) - H*np.cos(data - Hth_array))
                s_new = np.random.random_sample((N**2,))*2*np.pi # Proposed new spin array
                E1 = np.array(-J*(np.cos(s_new - s_left) + np.cos(s_new - s_right) + np.cos(s_new - s_up) + np.cos(s_new - s_down)) - H*np.cos(s_new - Hth_array))
                dE = E1 - E0
                vec = np.vectorize(self.rotate_one)
                data2 = vec(dE, data, s_new)
            
            self.data = data2
            self.update()
            root.update_idletasks()
            root.update()
            
            count = self.stepcount.get()
            count += 1
            self.stepcount.set(count)
        else:
            return
    
    def sys_step_metro(self, *args):
        ''' Evolves spin sites by stepping systematically through them '''
        model = self.model.get()
        
        if self.contupdate.get(): # Continuously updates lattice as each spin is flipped/not flipped
            j = 0
            N = self.N.get()
            
            # For each spin element
            while j < (N**2):
                if self.run == True:
                    data = np.array(self.data)
                    s2 = self.single_step(j)
                    
                    if model == "Ising Model":
                        element = data[j]
                        data[j] = s2*element
                    else:
                        data[j] = s2
                    
                    self.data = data
                    
                    # Update energy and/or magnetisation
                    if self.update_en == True:
                        self.E.append(np.sum(np.array(self.energy()))/(N**2))
                        
                    if self.update_mag == True:
                        self.M.append(np.sum(np.array(data))/(N**2))
                    
                    self.update()
                    root.update_idletasks()
                    root.update()
                    j += 1
                else:
                    return
            
            count = self.stepcount.get()
            count += 1
            self.stepcount.set(count)
        else: # Updates lattice only after all the spins have been run through
            j = 0
            N = self.N.get()
            
            if self.run == True:
                while j < (N**2):
                    data = np.array(self.data)
                    s2 = self.single_step(j)
                    
                    if model == "Ising Model":
                        element = data[j]
                        data[j] = s2*element
                    else:
                        data[j] = s2
                    
                    self.data = data
                    
                    # Update energy and/or magnetisation
                    if self.update_en == True:
                        self.E.append(np.sum(np.array(self.energy()))/(N**2))
                        
                    if self.update_mag == True:
                        self.M.append(np.sum(np.array(data))/(N**2))
                    
                    j += 1
                
                self.update()
                root.update_idletasks()
                root.update()
                
                count = self.stepcount.get()
                count += 1
                self.stepcount.set(count)
            else:
                return
    
    def ran_step_metro(self, *args):
        ''' Evolves spin sites by stepping randomly through them '''
        model = self.model.get()
        
        if self.contupdate.get(): # Continuously updates lattice as each spin is flipped/not flipped
            N = self.N.get()
            sequence = np.random.randint(N**2, size = N**2)
            j = 0
            
            # For each spin element in the random sequence
            while j < (N**2):
                if self.run == True:
                    data = np.array(self.data)
                    n = sequence[j] # Chooses spin to be flipped/not flipped
                    s2 = self.single_step(n)
                    
                    if model == "Ising Model":
                        element = data[n]
                        data[n] = s2*element
                    else:
                        data[n] = s2
                    
                    self.data = data
                    
                    # Update energy and/or magnetisation
                    if self.update_en == True:
                        self.E.append(np.sum(np.array(self.energy()))/(N**2))
                        
                    if self.update_mag == True:
                        self.M.append(np.sum(np.array(data))//(N**2))
                    
                    self.update()
                    root.update_idletasks()
                    root.update()
                    j += 1
                else:
                    return
            
            count = self.stepcount.get()
            count += 1
            self.stepcount.set(count)
        else: # Updates lattice only after all the spins have been run through
            N = self.N.get()
            sequence = np.random.randint(N**2, size = N**2)
            j = 0
            
            if self.run == True:
                while j < (N**2):
                    data = np.array(self.data)
                    n = sequence[j] # Chooses spin to be flipped/not flipped
                    s2 = self.single_step(n)
                    
                    if model == "Ising Model":
                        element = data[n]
                        data[n] = s2*element
                    else:
                        data[n] = s2
                    
                    self.data = data
                    
                    # Update energy and/or magnetisation
                    if self.update_en == True:
                        self.E.append(np.sum(np.array(self.energy()))/(N**2))
                        
                    if self.update_mag == True:
                        self.M.append(np.sum(np.array(data))/(N**2))
                    
                    j += 1
                
                self.update()
                root.update_idletasks()
                root.update()
                
                count = self.stepcount.get()
                count += 1
                self.stepcount.set(count)
            else:
                return
       
    def start_simulation(self, *args):
        ''' Runs simulation for given step count, without taking any measurements '''
        self.update_canvas()
        evo = self.evolution.get()
        
        if evo == "Simultaneous":
            self.run = True
            while self.run == True:
                self.sim_step_metro()
        elif evo == "Alternating":
            self.run = True
            while self.run == True:
                self.alt_step_metro()
        elif evo == "Systematic":
            self.run = True
            while self.run == True:
                self.sys_step_metro()
        elif evo == "Randomised":
            self.run = True
            while self.run == True:
                self.ran_step_metro()
        else:
            messagebox.showinfo("Error", "Error running simulation")
    
    def stop_simulation(self, *args):
        ''' Stops simulation '''
        self.run = False
    
    def reset_step(self, *args):
        ''' Resets step count '''
        self.stepcount.set(0)
    
    def intersect(self, array1, array2, N):
        ''' Finds the N-th intersection point between two arrays of equal length '''
        d_array = array1 - array2
        sign_array = np.sign(d_array)
        timelength = len(sign_array)
        i = 0
        j = 0 # Number of times intersection has occured
        k = timelength # If there is no intersection, return length of array
        
        while i < timelength - 1:
            if sign_array[i] == 0: # Zero crossing (intersection) - this is to avoid division by zero
                if j == N: # N-th intersection reached
                    k = i
                    break
                else:
                    j += 1
            else:
                change = sign_array[i+1]/sign_array[i]
                if change == -1: # Sign change (intersection)
                    if j == N: # N-th intersection reached
                        k = (d_array[i+1]/d_array[i]) - 1 + i
                        break
                    else:
                        j += 1
                else: # No intersection
                    i += 1
        return k
    
    def energy(self, *args):
        ''' Returns energy of each spin site, as an array '''
        N = self.N.get()
        J = self.J.get()
        H = self.H.get()
        model = self.model.get()
        
        data = np.array(self.data)
        
        # Re-arrange array such that indices refer to spins directly left/right/up/down
        s_left1 = data[-1]
        s_left2 = data[:-1]
        s_left = np.append(s_left1, s_left2)
        
        s_right1 = data[1:]
        s_right2 = data[0]
        s_right = np.append(s_right1, s_right2)
        
        s_up1 = data[-N:]
        s_up2 = data[:-N]
        s_up = np.append(s_up1, s_up2)
        
        s_down1 = data[N:]
        s_down2 = data[:N]
        s_down = np.append(s_down1, s_down2)
        
        if model == "Ising Model":
            E = -np.array(0.5*J*(np.multiply(data, s_left) + np.multiply(data, s_right) + np.multiply(data, s_up) + np.multiply(data, s_down)) + H*data)
            # Nota bene: halve interaction energy to avoid double-counting
        else:
            Hth = self.Hth.get()
            Hth_array = np.ones(N**2)*Hth
            E = np.array(-0.5*J*(np.cos(data - s_left) + np.cos(data - s_right) + np.cos(data - s_up) + np.cos(data - s_down)) - H*np.cos(data - Hth_array))
        
        return E
    
    '''
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     FUNCTIONS FOR (1)     ————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def therm(self, *args):
        ''' Creates new window for (1) Thermalise '''
        window1 = tk.Toplevel(root)
        window1.title("(1) Thermalise")
        
        window1.thermstart_button = ttk.Button(window1, text = "Start single", command = self.start_therm)
        window1.thermstop_button = ttk.Button(window1, text = "Stop and plot", command = self.stop_therm)
        window1.decay_label = ttk.Label(window1, text = "Decay plot")
        window1.decay_checkbox = ttk.Checkbutton(window1, variable = self.thermdecay)
        window1.temprange4_label = ttk.Label(window1, text = "Lower temperature bound")
        window1.temprange4_entry = ttk.Entry(window1, textvariable = self.temprange4)
        window1.temprange5_label = ttk.Label(window1, text = "Upper temperature bound")
        window1.temprange5_entry = ttk.Entry(window1, textvariable = self.temprange5)
        window1.temppoints_label = ttk.Label(window1, text = "No. of sampling points")
        window1.temppoints_entry = ttk.Entry(window1, textvariable = self.samp3)
        window1.steps_label = ttk.Label(window1, text = "Number of steps")
        window1.steps_entry = ttk.Entry(window1, textvariable = self.thermstep)
        window1.thermflips_label = ttk.Label(window1, text = "Thermalisation crossings")
        window1.thermflips_entry = ttk.Entry(window1, textvariable = self.thermflips)
        window1.magstart = ttk.Button(window1, text = "Compare thermalisations", command = self.therm_comp)
        
        window1.thermstart_button.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.thermstop_button.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.decay_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.decay_checkbox.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.temprange4_label.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.temprange4_entry.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.temprange5_label.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.temprange5_entry.grid(column = 1, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.temppoints_label.grid(column = 0, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.temppoints_entry.grid(column = 1, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.steps_label.grid(column = 0, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.steps_entry.grid(column = 1, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.thermflips_label.grid(column = 0, row = 6, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.thermflips_entry.grid(column = 1, row = 6, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window1.magstart.grid(column = 0, row = 7, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
    
    def start_therm(self, *args):
        ''' Runs thermalisation simulation '''
        self.update_canvas()
        evo = self.evolution.get()
        self.ave_energy = []
        self.E = []
        self.reset_step()
        
        # Run simulation
        self.run = True
        while self.run == True:
            if evo == "Simultaneous":
                self.sim_step_metro()
                # Record average energy
                E = np.array(self.energy())
                ave_E = np.mean(E)
                step = self.stepcount.get()
                addon = [step, ave_E]
                self.ave_energy.append(addon)
            elif evo == "Alternating":
                self.alt_step_metro()
                # Record average energy
                E = np.array(self.energy())
                ave_E = np.mean(E)
                step = self.stepcount.get()
                addon = [step, ave_E]
                self.ave_energy.append(addon)
            elif evo == "Systematic":
                self.update_en = True # Update energy with each spin change
                self.sys_step_metro()
            elif evo == "Randomised":
                self.update_en = True # Update energy with each spin change
                self.ran_step_metro()
            else:
                messagebox.showinfo("Error", "Error running simulation")
    
    def stop_therm(self, *args):
        ''' Stops thermalisation simulation and plots energy against step count '''
        self.run = False
        evo = self.evolution.get()
        N = self.N.get()
        
        if evo == "Simultaneous" or evo == "Alternating":
            energy_data = np.array(self.ave_energy)
            t = energy_data[:,0]
            E = energy_data[:,1]
        else:
            E = np.array(self.E)
            t = np.arange(len(E))/(N**2) # Normalise stepcount by grid size
        
        if self.thermdecay.get():
            tlen = len(t)
            # E_ave = np.mean(E[-np.int(np.rint(tlen/2)):]) # Set thermalised energy as average energy in final 1/2th of data
            E_ave = E[-1]
            E_scaled = np.absolute((E - E_ave*np.ones(tlen))/(E[0] - E_ave))
            
            # Plot average energy in the form of a logarithmic decay plot
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            plt.plot(t, E_scaled, 'k-')
            plt.xlabel("Step Count")
            plt.ylabel("Decay in off-equilibrium energy")
            plt.yscale('log')
            plt.title("Energy as a measure of thermalisation")
            plt.show()
        else:
            # Unmodified plot
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            plt.plot(t, E, 'k-')
            plt.xlabel("Step Count")
            plt.ylabel("Energy per site")
            plt.title("Energy as a measure of thermalisation")
            plt.show()
    
    def therm_comp(self, *args):
        ''' Investigates thermalisation times for different temperatures, defining thermalisation as first intersection with average energy '''
        # Get simulation parameters
        evo = self.evolution.get()
        temprange4 = self.temprange4.get()
        temprange5 = self.temprange5.get()
        samp3 = self.samp3.get()
        thermstep = self.thermstep.get()
        temp_diff = (temprange5 - temprange4)/(samp3 - 1)
        therm_data = []
        i = temprange4
        n = self.thermflips.get()
        N = self.N.get()
        start = time.time()
        
        # For each temperature
        while i < temprange5 + temp_diff/2:
            self.temp.set(i)
            self.update_canvas()
            self.ave_energy = []
            self.E = []
            self.set_lattice()
            self.reset_step()
            j = 0
            
            # Run simulation
            self.run = True
            
            while j < thermstep:
                if evo == "Simultaneous":
                    self.sim_step_metro()
                    # Record system energy
                    E = np.array(self.energy())
                    ave_E = np.mean(E)
                    step = self.stepcount.get()
                    addon = [step, ave_E]
                    self.ave_energy.append(addon)
                elif evo == "Alternating":
                    self.alt_step_metro()
                    # Record system energy
                    E = np.array(self.energy())
                    ave_E = np.mean(E)
                    step = self.stepcount.get()
                    addon = [step, ave_E]
                    self.ave_energy.append(addon)
                elif evo == "Systematic":
                    self.update_en = True # Update energy with each spin change
                    self.sys_step_metro()
                elif evo == "Randomised":
                    self.update_en = True # Update energy with each spin change
                    self.ran_step_metro()
                else:
                    messagebox.showinfo("Error", "Error running simulation")
                
                j += 1
            
            self.run = False
            energy_data = np.array(self.ave_energy)
            
            if evo == "Simultaneous" or evo == "Alternating":
                energy_data = np.array(self.ave_energy)
                E = np.array(energy_data[:,1])
            else:
                E = np.array(self.E)
            
            # Define thermalisation time as number of steps taken by energy offset from equilibrium to cross 0.1 deviation threshold N times
            length_E = len(E)
            E_ave = np.mean(E[-np.int(np.rint(length_E/2)):])
            E0 = E[0]
            E_scaled = (E - E_ave*np.ones(length_E))/(E0 - E_ave) # Re-scale initial offset = 1
            boundary = np.ones(length_E)*0.1
            
            if evo == "Simultaneous" or evo == "Alternating":
                therm_time = np.absolute(self.intersect(E_scaled, boundary, n))
            else:
                therm_time = np.absolute(self.intersect(E_scaled, boundary, n))/(N**2) # Normalise by grid size
            
            addon = [i, therm_time]
            therm_data.append(addon)
            i += temp_diff
        
        end = time.time()
        if self.runtime.get():
            timetaken = end - start
            print("Run time / s: " + str(timetaken))
        
        T = np.array(therm_data)[:,0]
        Th = np.array(therm_data)[:,1]
        
        # Plot
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.plot(T, Th, 'k.')
        plt.xlabel("Temperature")
        plt.ylabel("Thermalisation time")
        plt.title("Temperature-dependence of thermalisation time")
        plt.show()
    
    '''
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     FUNCTIONS FOR (2)     ————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def autocor(self, *args):
        ''' Runs simulation for a preset number of steps and plots autocorrelation against step count '''
        window2 = tk.Toplevel(root)
        window2.title("(2) Autocorrelation")
        
        window2.cor_label = ttk.Label(window2, text = "Time lag buffer")
        window2.cor_entry = ttk.Entry(window2, textvariable = self.tlagbuffer)
        window2.corstart_button = ttk.Button(window2, text = "Start single", command = self.start_cor)
        window2.corstop_button = ttk.Button(window2, text = "Stop and plot", command = self.stop_cor)
        window2.temprange4_label = ttk.Label(window2, text = "Lower temperature bound")
        window2.temprange4_entry = ttk.Entry(window2, textvariable = self.temprange6)
        window2.temprange5_label = ttk.Label(window2, text = "Upper temperature bound")
        window2.temprange5_entry = ttk.Entry(window2, textvariable = self.temprange7)
        window2.temppoints_label = ttk.Label(window2, text = "No. of sampling points")
        window2.temppoints_entry = ttk.Entry(window2, textvariable = self.samp4)
        window2.steps_label = ttk.Label(window2, text = "Number of steps")
        window2.steps_entry = ttk.Entry(window2, textvariable = self.corstep)
        window2.magstart = ttk.Button(window2, text = "Compare autocorrelations", command = self.comp_cor)
        
        window2.cor_label.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.cor_entry.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.corstart_button.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.corstop_button.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.temprange4_label.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.temprange4_entry.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.temprange5_label.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.temprange5_entry.grid(column = 1, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.temppoints_label.grid(column = 0, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.temppoints_entry.grid(column = 1, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.steps_label.grid(column = 0, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.steps_entry.grid(column = 1, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window2.magstart.grid(column = 0, row = 6, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        
    def start_cor(self, *args):
        ''' Runs autocorrelation simulation '''
        self.update_canvas()
        evo = self.evolution.get()
        self.mag = []
        self.M = []
        
        # Run simulation
        self.run = True
        while self.run == True:
            if evo == "Simultaneous":
                self.sim_step_metro()
                # Record magnetisation
                data = self.data
                M = np.sum(self.data)
                step = self.stepcount.get()
                addon = [step, M]
                self.mag.append(addon)
            elif evo == "Alternating":
                self.alt_step_metro()
                # Record magnetisation
                data = self.data
                M = np.sum(self.data)
                step = self.stepcount.get()
                addon = [step, M]
                self.mag.append(addon)
            elif evo == "Systematic":
                self.update_mag = True
                self.sys_step_metro()
            elif evo == "Randomised":
                self.update_mag = True
                self.ran_step_metro()
            else:
                messagebox.showinfo("Error", "Error running simulation")
    
    def stop_cor(self, *args):
        ''' Stops autocorrelation simulation and plots autocorrelation as a function of time lag '''
        self.run = False
        evo = self.evolution.get()
        model = self.model.get()
        N = self.N.get()
        
        if evo == "Simultaneous" or evo == "Alternating":
            data = np.array(self.mag)
            t = data[:,0]
            M = data[:,1]
            steps = len(M)
        else:
            data = np.array(self.M)
            M = data
            steps = len(M)
            t = (np.arange(steps))/(N**2)
        
        M_ave = np.mean(M)
        
        # Autocovariance at zero time lag
        if model == "Ising Model":
            M_prime = M - M_ave*np.ones((steps,), dtype = int)
            MM = np.multiply(M_prime, M_prime)
            buffer = self.tlagbuffer.get()
        else:
            M_prime = (M - M_ave*np.ones((steps,), dtype = float))/(2*np.pi)
            MM = np.cos(M_prime - M_prime)
            buffer = (self.tlagbuffer.get())*(N**2)
        
        self.A = []
        acov = np.mean(MM)
        addon = [0, acov]
        self.A.append(addon)
        i = 1
        
        # Determine autocovariance
        while i < steps - buffer:
            M1 = M_prime[:-i]
            M2 = M_prime[i:]
            
            if model == "Ising Model":
                MM = np.multiply(M1, M2)
            else:
                MM = np.cos(M1 - M2)
            
            acov = np.mean(MM)
            addon = [i, acov]
            self.A.append(addon)
            i += 1
        
        # Determine autocorrelation
        Acovar = np.array(self.A)[:,1]
        A0 = Acovar[0]
        Acorr = np.true_divide(Acovar, A0)
        
        if evo == "Simultaneous" or evo == "Alternating":
            T = np.arange(steps - buffer)
        else:
            T = (np.arange(len(Acorr)))/(N**2)
        
        # Plot
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.semilogy(T, Acorr, 'k-')
        plt.xlabel("Time lag")
        plt.ylabel("Autocorrelation")
        plt.title("Variation of autocorrelation with time lag")
        plt.show()
    
    def comp_cor(self, *args):
        ''' Investigates time lag (at which autocorrelation falls to 1/e) for different temperatures '''
        # Get simulation parameters
        temprange6 = self.temprange6.get()
        temprange7 = self.temprange7.get()
        samp4 = self.samp4.get()
        corstep = self.corstep.get()
        temp_diff = (temprange7 - temprange6)/(samp4 - 1)
        cor_data = []
        evo = self.evolution.get()
        model = self.model.get()
        k = temprange6
        start = time.time()
        
        # For each temperature
        while k < temprange7 + temp_diff/2:
            self.temp.set(k)
            self.update_canvas()
            self.set_lattice()
            self.reset_step()
            self.mag = []
            j = 0
            
            # Run simulation
            self.run = True
            while j < corstep:
                if evo == "Simultaneous":
                    self.sim_step_metro()
                elif evo == "Alternating":
                    self.alt_step_metro()
                elif evo == "Systematic":
                    self.sys_step_metro()
                elif evo == "Randomised":
                    self.ran_step_metro()
                else:
                    messagebox.showinfo("Error", "Error running simulation")
                
                data = self.data
                M = np.sum(self.data)
                step = self.stepcount.get()
                addon = [step, M]
                self.mag.append(addon)
                j += 1
            
            self.run = False
            
            # Determine fall-off in autocorrelation with time lag
            # Analogous to previous function
            data = np.array(self.mag)
            t = data[:,0]
            M = data[:,1]
            steps = len(t)
            M_ave = np.mean(M)
            
            if model == "Ising Model":
                M_prime = M - M_ave*np.ones((steps,), dtype = int)
                MM = np.multiply(M_prime, M_prime)
            else:
                M_prime = (M - M_ave*np.ones((steps,), dtype = float))/(2*np.pi)
                MM = np.cos(M_prime - M_prime)
            
            buffer = self.tlagbuffer.get()
            self.A = []
            acov = np.mean(MM)
            addon = [0, acov]
            self.A.append(addon)
            
            i = 1
            while i < steps - buffer:
                M1 = M_prime[:-i]
                M2 = M_prime[i:]
                
                if model == "Ising Model":
                    MM = np.multiply(M1, M2)
                else:
                    MM = np.cos(M1 - M2)
                
                acov = np.mean(MM)
                addon = [i, acov]
                self.A.append(addon)
                i += 1
            
            Acovar = np.array(self.A)[:,1]
            length = len(Acovar)
            A0 = Acovar[0]
            
            if A0 == 0:
                Acorr = np.empty(length)
                Acorr[:] = np.nan
            else:
                Acorr = np.array(np.true_divide(Acovar, A0))
            
            # Time taken for autocorrelation to fall by 1/e
            decay_array = np.ones(length)/np.e
            intersect = np.absolute(self.intersect(Acorr, decay_array, 1))
            addon = [k, intersect]
            cor_data.append(addon)
            k += temp_diff
        
        end = time.time()
        if self.runtime.get():
            timetaken = end - start
            print("Run time / s: " + str(timetaken))
        
        T = np.array(cor_data)[:,0]
        tlag = np.array(cor_data)[:,1]
        
        # Plot
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.plot(T, tlag, 'k.')
        plt.xlabel("Temperature")
        plt.ylabel("Time Lag")
        plt.title("Time lag for autocorrelation to fall by $e$")
        plt.show()
    
    '''
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     FUNCTIONS FOR (3)     ————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def meanmag(self, *args):
        ''' Creates new window for (3) Mean Magnetisation '''
        window3 = tk.Toplevel(root)
        window3.title("(3) Mean Magnetisation")
        
        window3.temprange0_label = ttk.Label(window3, text = "Lower temperature bound")
        window3.temprange0_entry = ttk.Entry(window3, textvariable = self.temprange0)
        window3.temprange_label = ttk.Label(window3, text = "Upper temperature bound")
        window3.temprange_entry = ttk.Entry(window3, textvariable = self.temprange)
        window3.temppoints_label = ttk.Label(window3, text = "No. of sampling points")
        window3.temppoints_entry = ttk.Entry(window3, textvariable = self.samp)
        window3.thermbuffer_label = ttk.Label(window3, text = "Thermalisation buffer time")
        window3.thermbuffer_entry = ttk.Entry(window3, textvariable = self.tbuffer)
        window3.steps_label = ttk.Label(window3, text = "Number of steps")
        window3.steps_entry = ttk.Entry(window3, textvariable = self.magstep)
        window3.magstart = ttk.Button(window3, text = "Start", command = self.start_mag)
        
        window3.temprange0_label.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.temprange0_entry.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.temprange_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.temprange_entry.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.temppoints_label.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.temppoints_entry.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.thermbuffer_label.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.thermbuffer_entry.grid(column = 1, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.steps_label.grid(column = 0, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.steps_entry.grid(column = 1, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window3.magstart.grid(column = 0, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        
    def start_mag(self, *args):
        ''' Runs magnetisation simulation simulation for a preset number of steps and temperature range, then plots mean magnetisation against temperature '''
        # Get simulation parameters
        self.update_canvas()
        evo = self.evolution.get()
        temprange0 = self.temprange0.get()
        temprange = self.temprange.get()
        samp = self.samp.get()
        tbuffer = self.tbuffer.get()
        magstep = self.magstep.get()
        model = self.model.get()
        start = time.time()
        
        # Run simulation
        self.run = True
        istep = (temprange - temprange0)/(samp - 1) # Temperature step
        M = [] # Holder for main data
        
        while self.run == True:
            i = temprange0
            
            # For each temperature
            while i < temprange + istep/2:
                self.temp.set(i)
                self.set_lattice()
                self.reset_step()

                # Buffer
                j = 0
                while j < tbuffer:
                    if evo == "Simultaneous":
                        self.sim_step_metro()
                    elif evo == "Alternating":
                        self.alt_step_metro()
                    elif evo == "Systematic":
                        self.sys_step_metro()
                    elif evo == "Randomised":
                        self.ran_step_metro()
                    else:
                        messagebox.showinfo("Error", "Error running simulation")
                    j += 1
                
                # Magnetisation measurements
                k = 0
                Mabs = [] # Array for absolute total magnetisation
                
                while k < magstep:
                    if evo == "Simultaneous":
                        self.sim_step_metro()
                    elif evo == "Alternating":
                        self.alt_step_metro()
                    elif evo == "Systematic":
                        self.sys_step_metro()
                    elif evo == "Randomised":
                        self.ran_step_metro()
                    else:
                        messagebox.showinfo("Error", "Error running simulation")
                    
                    data = np.array(self.data)
                    if model == "Ising Model":
                        m_sum = np.sum(data)
                    else:
                        m_sum = complex(np.sum(np.cos(data)), np.sum(np.sin(data))) # Encodes total magnetisation vector as a complex number
                    Mabs.append(m_sum)
                    k += 1
                
                if model == "Ising Model":
                    Mave = np.absolute(np.mean(Mabs))
                else:
                    Mave = abs(np.mean(Mabs))
                
                addon = [i, Mave]
                M.append(addon)
                i += istep
            
            self.run = False
        
        end = time.time()
        if self.runtime.get():
            timetaken = end - start
            print("Run time/s: " + str(timetaken))
        
        T = np.array(M)[:,0]
        Mean_mag = np.array(M)[:,1]
        
        if model == "Ising Model":
            # C.N. Yang's analytical solution
            trunc_T = np.linspace(T[1], 2.26918, num = 5001)
            T_inv = np.reciprocal(trunc_T)
            J = self.J.get()
            c = Mean_mag[1]*((1 - (np.sinh(2*J*T_inv[1]))**-4)**-0.125)
            theo = c*((1 - (np.sinh(2*J*T_inv))**-4)**0.125)
            T_pad = np.concatenate((trunc_T, [2.2691853]), axis = 0)
            theo_pad = np.concatenate((theo, [0]), axis = 0)
            
            # Plot
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            plt.plot(T_pad, theo_pad, 'b-', label = "Theoretical prediction")
            plt.plot(T, Mean_mag, 'k-', label = "Computational Result")
            plt.xlabel("Temperature")
            plt.ylabel("Mean absolute magnetisation")
            plt.title("Temperature-dependence of mean magnetisation")
            plt.legend()
            plt.show()
        else:
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            plt.plot(T, Mean_mag, 'k-', label = "Computational Result")
            plt.xlabel("Temperature")
            plt.ylabel("Mean absolute magnetisation")
            plt.title("Temperature-dependence of mean magnetisation")
            plt.legend()
            plt.show()
    
    '''
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     FUNCTIONS FOR (4)     ————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def heatcap(self, *args):
        ''' Creates new window for (4) Heat Capacity '''
        window4 = tk.Toplevel(root)
        window4.title("(4) Heat Capacity")
        
        window4.temprange2_label = ttk.Label(window4, text = "Lower temperature")
        window4.temprange2_entry = ttk.Entry(window4, textvariable = self.temprange2)
        window4.temprange3_label = ttk.Label(window4, text = "Upper temperature")
        window4.temprange3_entry = ttk.Entry(window4, textvariable = self.temprange3)
        window4.temppoints_label = ttk.Label(window4, text = "No. of sampling points")
        window4.temppoints_entry = ttk.Entry(window4, textvariable = self.samp2)
        window4.thermbuffer_label = ttk.Label(window4, text = "Thermalisation buffer time")
        window4.thermbuffer_entry = ttk.Entry(window4, textvariable = self.tbuffer2)
        window4.steps_label = ttk.Label(window4, text = "Number of steps")
        window4.steps_entry = ttk.Entry(window4, textvariable = self.capstep)
        window4.capstart = ttk.Button(window4, text = "Start", command = self.start_heatcap)
        
        window4.temprange2_label.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.temprange2_entry.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.temprange3_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.temprange3_entry.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.temppoints_label.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.temppoints_entry.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.thermbuffer_label.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.thermbuffer_entry.grid(column = 1, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.steps_label.grid(column = 0, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.steps_entry.grid(column = 1, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window4.capstart.grid(column = 0, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        
    def start_heatcap(self, *args):
        ''' Runs simulation for a preset number of steps and temperature range, then plots heat capacity against temperature
            Here we take critical temperature as the weighted mean of five highest points '''
        # Get simulation parameters
        self.update_canvas()
        evo = self.evolution.get()
        temprange2 = self.temprange2.get()
        temprange3 = self.temprange3.get()
        samp = self.samp2.get()
        tbuffer = self.tbuffer2.get()
        capstep = self.capstep.get()
        N = self.N.get()
        start = time.time()
        
        # Run simulation
        self.run = True
        istep = (temprange3 - temprange2)/(samp - 1) # Temperature step
        C = [] # Holder for main data
        
        while self.run == True:
            i = temprange2
            while i < temprange3 + istep/2: # For each temperature
                self.temp.set(i)
                self.set_lattice()
                self.reset_step()
                
                # Thermalisation buffer
                j = 0
                while j < tbuffer:
                    if evo == "Simultaneous":
                        self.sim_step_metro()
                    elif evo == "Alternating":
                        self.alt_step_metro()
                    elif evo == "Systematic":
                        self.sys_step_metro()
                    elif evo == "Randomised":
                        self.ran_step_metro()
                    else:
                        messagebox.showinfo("Error", "Error running simulation")
                    j += 1
                
                # Heat capacity
                k = 0
                E = [] # Array for system energy
                
                while k < capstep:
                    if evo == "Simultaneous":
                        self.sim_step_metro()
                    elif evo == "Alternating":
                        self.alt_step_metro()
                    elif evo == "Systematic":
                        self.sys_step_metro()
                    elif evo == "Randomised":
                        self.ran_step_metro()
                    else:
                        messagebox.showinfo("Error", "Error running simulation")
                    
                    E_array = np.array(self.energy())
                    Ein = np.sum(E_array)
                    E.append(Ein)
                    k += 1
                
                # Determines heat capacity from standard deviation in energy
                if i == 0:
                    cap = 0
                else:
                    cap = ((np.std(E))**2)/(i**2)
                
                addon = [i, cap]
                C.append(addon)
                i += istep
                
            self.run = False
        
        end = time.time()
        if self.runtime.get():
            timetaken = end - start
            print("Run time / s: " + str(timetaken))
        
        T = np.array(C)[:,0]
        heat_cap = np.array(C)[:,1]
        product = np.multiply(T, heat_cap)
        
        # Pick out 5 largest heat capacities and finds their weighted mean
        heat_cap_sort = np.argsort(heat_cap)
        largest = np.array(heat_cap_sort)[-5:]
        n = 0
        heat_cap_max = []
        product_max = []
        
        while n < 5:
            p = largest[n]
            heat_cap_max.append(heat_cap[p])
            product_max.append(product[p])
            n += 1
        
        sum_product = np.sum(product_max)
        sum_heat_cap = np.sum(heat_cap_max)
        mean = np.round_(sum_product/sum_heat_cap, decimals = 3)
        
        # Analytical solution
        Tc = 2.269185
        J = self.J.get()
        prefac = (2/np.pi)*((2*J/Tc)**2)
        constant = np.log(2*Tc/J) - 1 - (np.pi/4)
        T2min = T[0]
        T2max = T[-1]
        T2 = np.linspace(T2min, T2max, num = 2001)
        length = len(T2)
        logfac = np.absolute(np.ones(length) - (T2/Tc))
        Nsq = N**2
        theo = Nsq*prefac*(np.ones(length)*constant - np.log(logfac))
        max = np.amax(heat_cap)*1.05
        
        # Plot
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.plot(T, heat_cap, 'k.', label = "Computational result")
        plt.plot(T2, theo, 'b-', label = "Theoretical result")
        plt.xlabel("Temperature")
        plt.ylabel("Heat Capacity")
        axes = plt.gca()
        axes.set_ylim([0, max])
        plt.title("Temperature-dependence of heat capacity\n" + "Estimated critical temperature = " + str(mean))
        plt.legend()
        plt.show()
    
    '''
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     FUNCTIONS FOR (5)     ————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def fs_scaling(self, *args):
        ''' Creates new window for (5) Finite-Size Scaling '''
        self.heatcap()
        
        window5 = tk.Toplevel(root)
        window5.title("(5) Finite-Size Scaling")
        
        window5.Nlower_label = ttk.Label(window5, text = "Lower N")
        window5.Nlower_entry = ttk.Entry(window5, textvariable = self.Ndown)
        window5.Nupper_label = ttk.Label(window5, text = "Upper N")
        window5.Nupper_entry = ttk.Entry(window5, textvariable = self.Nup)
        window5.Nvalues_label = ttk.Label(window5, text = "Number of N samples")
        window5.Nvalues_entry = ttk.Entry(window5, textvariable = self.Nnum)
        window5.fsstart = ttk.Button(window5, text = "Investigate Scaling", command = self.fs_start)
        
        window5.Nlower_label.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window5.Nlower_entry.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window5.Nupper_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window5.Nupper_entry.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window5.Nvalues_label.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window5.Nvalues_entry.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window5.fsstart.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
    
    def fs_start(self, *args):
        ''' Determines critical temperature for a range of N and matches the relation to Onsager's result '''
        # Get simulation parameters
        Ndown = self.Ndown.get()
        Nup = self.Nup.get()
        Nnum = self.Nnum.get()
        ln_array = np.linspace(np.log(Ndown), np.log(Nup), num = Nnum)
        N_array_raw = np.exp(ln_array)
        N_array_raw2 = np.rint((N_array_raw - 1)/2)
        N_array = (2*N_array_raw2 + 1).astype(int) # N values rounded to nearest odd integer
        fs_data = []
        l = 0
        start = time.time()
        
        # For each lattice size
        while l < Nnum:
            Nnow = N_array[l]
            self.N.set(Nnow)
            self.set_lattice()
            self.update_canvas()
            evo = self.evolution.get()
            temprange2 = self.temprange2.get()
            temprange3 = self.temprange3.get()
            samp = self.samp2.get()
            tbuffer = self.tbuffer2.get()
            capstep = self.capstep.get()
            model = self.model.get()
            
            # Run simulation
            self.run = True
            istep = (temprange3 - temprange2)/(samp - 1) # Temperature step
            C = [] # Holder for main data
            while self.run == True:
                i = temprange2
                while i < temprange3 + istep/2: # For each temperature
                    self.temp.set(i)
                    self.set_lattice()
                    self.reset_step()
                    
                    # Thermalisation buffer
                    j = 0
                    while j < tbuffer:
                        if evo == "Simultaneous":
                            self.sim_step_metro()
                        elif evo == "Alternating":
                            self.alt_step_metro()
                        elif evo == "Systematic":
                            self.sys_step_metro()
                        elif evo == "Randomised":
                            self.ran_step_metro()
                        else:
                            messagebox.showinfo("Error", "Error running simulation")
                        j += 1
                    
                    # Heat capacity measurement
                    k = 0
                    E = [] # Array for system energy
                    
                    while k < capstep:
                        if evo == "Simultaneous":
                            self.sim_step_metro()
                        elif evo == "Alternating":
                            self.alt_step_metro()
                        elif evo == "Systematic":
                            self.sys_step_metro()
                        elif evo == "Randomised":
                            self.ran_step_metro()
                        else:
                            messagebox.showinfo("Error", "Error running simulation")
                        E_array = np.array(self.energy())
                        Ein = np.sum(E_array)
                        E.append(Ein)
                        k += 1
                    
                    if i == 0:
                        cap = 0
                    else:
                        cap = ((np.std(E))**2)/(i**2)
                    
                    addon = [i, cap]
                    C.append(addon)
                    i += istep
                
                self.run = False
            
            # Calculate weighted mean
            T = np.array(C)[:,0]
            heat_cap = np.array(C)[:,1]
            product = np.multiply(T, heat_cap)
            
            # Pick out 5 largest heat capacities and finds their weighted mean
            heat_cap_sort = np.argsort(heat_cap)
            largest = np.array(heat_cap_sort)[-5:]
            n = 0
            heat_cap_max = []
            product_max = []
            T_max = []
            while n < 5:
                p = largest[n]
                heat_cap_max.append(heat_cap[p])
                product_max.append(product[p])
                T_max.append(T[p])
                n += 1
            sum_product = np.sum(product_max)
            sum_heat_cap = np.sum(heat_cap_max)
            mean = sum_product/sum_heat_cap
            
            # Calculate standard deviation
            mean_array = np.ones(5)*mean
            numerator = np.sum(heat_cap_max*((T_max - mean_array)**2))
            denominator = (4/5)*np.sum(heat_cap_max)
            std = (numerator/denominator)**0.5
            addon = [Nnow, mean, std]
            fs_data.append(addon)
            l += 1
        
        end = time.time()
        if self.runtime.get():
            timetaken = end - start
            print("Run time / s: " + str(timetaken))
        
        if model == "Ising Model":
            # Fit curve to data
            data = np.array(fs_data)
            N_array = np.array(data[:,0])
            fs_array = np.array(data[:,1])
            std_array = np.array(data[:,2])
            def func(N, Tinf, a, nu):
                return Tinf + a*(N**(-1/nu))
            popt, pcov = curve_fit(func, N_array, fs_array, bounds = (0, [2.5, 100000, 10]), sigma = std_array)
            fit = func(N_array, *popt)
            Tinf_return = np.round_(popt[0], decimals = 3)
            a_return = np.round_(popt[1], decimals = 3)
            nu_return = np.round_(popt[2], decimals = 3)
            
            # Plot
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            plt.errorbar(N_array, fs_array, yerr = std_array, linewidth = 0.5, capsize = 2, capthick = 1, fmt = "none", label = "Error")
            plt.xlabel("Lattice size $N$")
            plt.ylabel("Critical temperature $T_C$")
            plt.title("Asymptotic approach of critical temperature to Onsager limit")
            plt.plot(N_array, fit, 'k', label = "Fitted curve")
            plt.plot(N_array, fs_array, 'k.', label = "Data")
            plt.text(0.75, 0.65, r"$T_C$(inf) = " + str(Tinf_return) + "\n" + r"$a$ = " + str(a_return) + "\n" + r"$\nu$ = " + str(nu_return), transform = plt.gca().transAxes)
            plt.legend()
            plt.show()
        else:
            data = np.array(fs_data)
            N_array = np.array(data[:,0])
            fs_array = np.array(data[:,1])
            std_array = np.array(data[:,2])
            
            # Plot
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            plt.errorbar(N_array, fs_array, yerr = std_array, linewidth = 0.5, capsize = 2, capthick = 1, fmt = "none", label = "Error")
            plt.xlabel("Lattice size $N$")
            plt.ylabel("Critical temperature $T_C$")
            plt.title("Critical temperature scaling")
            plt.plot(N_array, fs_array, 'k.', label = "Data")
            plt.legend()
            plt.show()
    
    '''
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     FUNCTIONS FOR (6)     ————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def hys(self, *args):
        ''' Creates new window for (6) Hysteresis '''
        window6 = tk.Toplevel(root)
        window6.title("(6) Hysteresis")
        
        window6.Hperiod_label = ttk.Label(window6, text = "H period")
        window6.Hperiod_entry = ttk.Entry(window6, textvariable = self.Hperiod)
        window6.Hamplitude_label = ttk.Label(window6, text = "H amplitude")
        window6.Hamplitude_entry = ttk.Entry(window6, textvariable = self.Hamp)
        window6.H_start = ttk.Button(window6, text = "Start", command = self.H_start)
        window6.H_stop = ttk.Button(window6, text = "Stop and plot", command = self.H_stop)
        
        window6.Hperiod_label.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window6.Hperiod_entry.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window6.Hamplitude_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window6.Hamplitude_entry.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window6.H_start.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window6.H_stop.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
    
    def H_start(self, *args):
        ''' Starts simulation, varying applied H and investigates hysteresis properties of system '''
        # Get simulation parameters
        period = self.Hperiod.get()
        amp = np.absolute(self.Hamp.get())
        H_up = np.linspace(0, amp, np.int(np.rint(period/4)), endpoint = False)
        H_down = np.linspace(amp, -amp, np.int(np.rint(period/2)), endpoint = False)
        H_back = np.linspace(-amp, 0, np.int(np.rint(period/4)), endpoint = False)
        H_1 = np.concatenate((H_up, H_down), axis = None)
        H_array = np.concatenate((H_1, H_back), axis = None)
        self.update_canvas()
        evo = self.evolution.get()
        self.mag2 = []
        model = self.model.get()
        therm_time = 10
        
        # Run simulation
        self.run = True
        while self.run == True:
            j = 0
            
            # Thermalisation buffer
            while j < therm_time:
                if evo == "Simultaneous":
                    self.sim_step_metro()
                elif evo == "Alternating":
                    self.alt_step_metro()
                elif evo == "Systematic":
                    self.sys_step_metro()
                elif evo == "Randomised":
                    self.ran_step_metro()
                else:
                    messagebox.showinfo("Error", "Error running simulation")
                j += 1
            
            # Hysteresis
            while self.run == True:
                i = 0
                while i < period:
                    H = H_array[i]
                    self.H.set(H)
                    
                    if evo == "Simultaneous":
                        self.sim_step_metro()
                    elif evo == "Alternating":
                        self.alt_step_metro()
                    elif evo == "Systematic":
                        self.sys_step_metro()
                    elif evo == "Randomised":
                        self.ran_step_metro()
                    else:
                        messagebox.showinfo("Error", "Error running simulation")
                    
                    data = self.data
                    
                    if model == "Ising Model":
                        M = np.sum(data)
                    else:
                        M = abs(complex(np.sum(np.cos(data)), np.sum(np.sin(data)))) # Absolute value of a vector: does not encode spin direction
                    
                    addon = [H, M]
                    self.mag2.append(addon)
                    i += 1
    
    def H_stop(self, *args):
        ''' Stops hysteresis simulation and plots '''
        self.run = False
        mag = self.mag2
        H = np.array(mag)[:,0]
        M = np.array(mag)[:,1]
        T = self.temp.get()
        
        # Plot
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.plot(H, M, 'k-')
        plt.xlabel("$H$")
        plt.ylabel("$M$")
        plt.title("Hysteresis loop at temperature = " + str(T))
        plt.show()
    
    '''
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     FUNCTIONS FOR (7)     ————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def corlen(self, *args):
        ''' Creates new window for (7) Correlation Length '''
        window7 = tk.Toplevel(root)
        window7.title("(7) Correlation Length")
        
        window7.start_button = ttk.Button(window7, text = "Start single", command = self.corlenstart)
        window7.stop_button = ttk.Button(window7, text = "Stop and plot", command = self.corlenstop)
        window7.decay_label = ttk.Label(window7, text = "Decay plot")
        window7.decay_checkbox = ttk.Checkbutton(window7, variable = self.decay7)
        window7.temprange8_label = ttk.Label(window7, text = "Lower temperature")
        window7.temprange8_entry = ttk.Entry(window7, textvariable = self.temprange8)
        window7.temprange9_label = ttk.Label(window7, text = "Upper temperature")
        window7.temprange9_entry = ttk.Entry(window7, textvariable = self.temprange9)
        window7.temppoints_label = ttk.Label(window7, text = "No. of sampling points")
        window7.temppoints_entry = ttk.Entry(window7, textvariable = self.samp5)
        window7.thermbuffer_label = ttk.Label(window7, text = "Thermalisation buffer time")
        window7.thermbuffer_entry = ttk.Entry(window7, textvariable = self.tbuffer3)
        window7.steps_label = ttk.Label(window7, text = "Number of steps")
        window7.steps_entry = ttk.Entry(window7, textvariable = self.corlenstep)
        window7.repeat_label = ttk.Label(window7, text = "Repetition")
        window7.repeat_entry = ttk.Entry(window7, textvariable = self.rep)
        window7.corlenstart = ttk.Button(window7, text = "Start", command = self.start_corlen)
        
        window7.start_button.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.stop_button.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.decay_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.decay_checkbox.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.temprange8_label.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.temprange8_entry.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.temprange9_label.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.temprange9_entry.grid(column = 1, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.temppoints_label.grid(column = 0, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.temppoints_entry.grid(column = 1, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.thermbuffer_label.grid(column = 0, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.thermbuffer_entry.grid(column = 1, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.steps_label.grid(column = 0, row = 6, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.steps_entry.grid(column = 1, row = 6, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.repeat_label.grid(column = 0, row = 7, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.repeat_entry.grid(column = 1, row = 7, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window7.corlenstart.grid(column = 0, row = 8, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
    
    def corlenstart(self, *args):
        ''' Runs correlation length simulation for a preset number of steps '''
        # Get simulation parameters
        self.update_canvas()
        evo = self.evolution.get()
        self.corl = []
        tbuffer = self.tbuffer3.get()
        self.reset_step()
        N = self.N.get()
        Nsq = N**2
        model = self.model.get()
        
        # Thermalisation buffer
        self.run = True
        i = 0
        while i < tbuffer:
            if evo == "Simultaneous":
                self.sim_step_metro()
            elif evo == "Alternating":
                self.alt_step_metro()
            elif evo == "Systematic":
                self.sys_step_metro()
            elif evo == "Randomised":
                self.ran_step_metro()
            else:
                messagebox.showinfo("Error", "Error running simulation")
            i += 1
        
        # Measurements
        while self.run == True:
            if evo == "Simultaneous":
                self.sim_step_metro()
            elif evo == "Alternating":
                self.alt_step_metro()
            elif evo == "Systematic":
                self.sys_step_metro()
            elif evo == "Randomised":
                self.ran_step_metro()
            else:
                messagebox.showinfo("Error", "Error running simulation")
            
            data = self.data
            corrfunc = []
            
            # Determine equal-time correlation
            n = 0
            while n < N/2:
                s0 = data[0::Nsq]
                sn = data[n::Nsq]
                
                if model == "Ising Model":
                    g0 = np.mean(np.multiply(s0, sn))
                    g1 = np.mean(s0)
                    g2 = np.mean(sn)
                    corrfunc.append([g0, g1, g2])
                else:
                    g0 = np.mean(np.cos(s0 - sn)) #
                    corrfunc.append(g0)
                n += 1
            
            self.corl.append(corrfunc)
                
    def corlenstop(self, *args):
        ''' Stops correlation length simulation and plots equal-time correlation function '''
        self.run = False
        cor = np.array(self.corl)
        model = self.model.get()
        
        if model == "Ising Model":
            # axis0 = time
            # axis1 = n
            # axis2 = g
            
            g0_data = cor[:,:,0]
            g1_data = cor[:,:,1]
            g2_data = cor[:,:,2]
            
            cor1 = np.mean(g0_data, axis = 0)
            cor2 = np.mean(g1_data, axis = 0)
            cor3 = np.mean(g2_data, axis = 0)
            
            correlation = cor1 - np.multiply(cor2, cor3)
            length = len(correlation)
            r = np.arange(length)
            
            # Plot
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            if self.decay7.get():
                plt.loglog(np.absolute(r[1:]), np.absolute(correlation[1:]), 'k-')
                plt.grid(b = True, which = 'both')
            else:
                plt.plot(r, correlation, 'k-')
            plt.title("Correlation function")
            plt.xlabel("Distance")
            plt.ylabel("Equal-time correlation function")
            plt.show()
        else:
            correlation = np.mean(cor, axis = 0)
            length = len(correlation)
            r = np.arange(length)
            
            # Plot
            plt.rc('text', usetex = True)
            plt.rc('font', family = 'serif')
            if self.decay7.get():
                plt.loglog(np.absolute(r[1:]), np.absolute(correlation[1:]), 'k-')
                plt.grid(b = True, which = 'both')
            else:
                plt.plot(r, correlation, 'k-')
            plt.title("Correlation function")
            plt.xlabel("Distance")
            plt.ylabel("Two-point correlation function")
            plt.show()
        
    def start_corlen(self, *args):
        ''' Runs correlation length simulation for a preset number of steps and temperature range, then plots correlation length against temperature '''
        # Get simulation parameters
        temprange8 = self.temprange8.get()
        temprange9 = self.temprange9.get()
        samp5 = self.samp5.get()
        tbuffer3 = self.tbuffer3.get()
        corlenstep = self.corlenstep.get()
        rep = self.rep.get()
        N = self.N.get()
        model = self.model.get()
        Nsq = N**2
        evo = self.evolution.get()
        Cor = []
        m = 0
        start = time.time
        
        # For each repetition
        while m < rep:
            # Run simulation
            self.run = True
            istep = (temprange9 - temprange8)/(samp5 - 1) # Temperature step
            C = [] # Holder for main data
            
            while self.run == True:
                i = temprange8
                
                # For each temperature
                while i < temprange9 + istep/2:
                    self.temp.set(i)
                    self.set_lattice()
                    self.reset_step()
                    
                    self.corl = []
                    
                    # Thermalisation buffer
                    j = 0
                    while j < tbuffer3:
                        if evo == "Simultaneous":
                            self.sim_step_metro()
                        elif evo == "Alternating":
                            self.alt_step_metro()
                        elif evo == "Systematic":
                            self.sys_step_metro()
                        elif evo == "Randomised":
                            self.ran_step_metro()
                        else:
                            messagebox.showinfo("Error", "Error running simulation")
                        j += 1
                    
                    # Correlation length
                    k = 0
                    while k < corlenstep:
                        if evo == "Simultaneous":
                            self.sim_step_metro()
                        elif evo == "Alternating":
                            self.alt_step_metro()
                        elif evo == "Systematic":
                            self.sys_step_metro()
                        elif evo == "Randomised":
                            self.ran_step_metro()
                        else:
                            messagebox.showinfo("Error", "Error running simulation")
                        
                        data = self.data
                        corrfunc = []
                        
                        # Analogous to before
                        n = 0
                        while n < N/2:
                            s0 = data[0::Nsq]
                            sn = data[n::Nsq]
                            
                            if model == "Ising Model":
                                g0 = np.mean(np.multiply(s0, sn))
                                g1 = np.mean(s0)
                                g2 = np.mean(sn)
                                corrfunc.append([g0, g1, g2])
                            else:
                                g0 = np.mean(np.cos(s0 - sn))
                                corrfunc.append(g0)
                            n += 1
                        
                        self.corl.append(corrfunc)
                        k += 1
                    
                    cor = np.array(self.corl)
                    
                    if model == "Ising Model":
                        g0_data = cor[:,:,0]
                        g1_data = cor[:,:,1]
                        g2_data = cor[:,:,2]
                        
                        cor1 = np.mean(g0_data, axis = 0)
                        cor2 = np.mean(g1_data, axis = 0)
                        cor3 = np.mean(g2_data, axis = 0)
                        
                        correlation = cor1 - np.multiply(cor2, cor3)
                    else:
                        correlation = np.mean(cor, axis = 0)
                    
                    length = len(correlation)
                    decay = np.ones(length)*np.exp(-1)
                    corlength = np.absolute(self.intersect(correlation, decay, 1))
                    addon = [i, corlength]
                    C.append(addon)
                    i += istep
                
                self.run = False
            
            T = np.array(C)[:,0]
            correlation = np.array(C)[:,1]
            Cor.append(correlation)
            m += 1
        
        end = time.time()
        if self.runtime.get():
            timetaken = end - start
            print("Run time / s: " + str(timetaken))
        
        ave_cor = np.mean(np.array(Cor), axis = 0)
        
        # Plot
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.plot(T, ave_cor, 'k.')
        plt.title("Temperature-dependence of correlation length")
        plt.show()
    
    '''
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    ————————————————————————————————————————————————————     FUNCTIONS FOR (8)     ————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    
    def sus(self, *args):
        ''' Creates new window for (8) Magnetic Susceptibility '''
        window8 = tk.Toplevel(root)
        window8.title("(8) Magnetic Susceptibility")
        
        window8.temprange10_label = ttk.Label(window8, text = "Lower temperature")
        window8.temprange10_entry = ttk.Entry(window8, textvariable = self.temprange10)
        window8.temprange11_label = ttk.Label(window8, text = "Upper temperature")
        window8.temprange11_entry = ttk.Entry(window8, textvariable = self.temprange11)
        window8.temppoints_label = ttk.Label(window8, text = "No. of sampling points")
        window8.temppoints_entry = ttk.Entry(window8, textvariable = self.samp6)
        window8.tbuffer4_label = ttk.Label(window8, text = "Thermalisation buffer")
        window8.tbuffer4_entry = ttk.Entry(window8, textvariable = self.tbuffer4)
        window8.steps_label = ttk.Label(window8, text = "Number of steps")
        window8.steps_entry = ttk.Entry(window8, textvariable = self.susstep)
        window8.sus_start = ttk.Button(window8, text = "Start", command = self.susstart)
        
        window8.temprange10_label.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.temprange10_entry.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.temprange11_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.temprange11_entry.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.temppoints_label.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.temppoints_entry.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.tbuffer4_label.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.tbuffer4_entry.grid(column = 1, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.steps_label.grid(column = 0, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.steps_entry.grid(column = 1, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window8.sus_start.grid(column = 0, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
    
    def susstart(self, *args):
        ''' Runs simulation for a range of temperatures, and plots magnetic susceptibility against temperature '''
        # Get simulation parameters
        self.update_canvas()
        evo = self.evolution.get()
        temprange10 = self.temprange10.get()
        temprange11 = self.temprange11.get()
        samp6 = self.samp6.get()
        tbuffer4 = self.tbuffer4.get()
        susstep = self.susstep.get()
        model = self.model.get()
        start = time.time()
        
        # Run simulation
        self.run = True
        istep = (temprange11 - temprange10)/(samp6 - 1) # Temperature step
        S = [] # Holder for main data
        
        while self.run == True:
            i = temprange10
            
            # For each temperature
            while i < temprange11 + istep/2:
                self.temp.set(i)
                self.set_lattice()
                self.reset_step()
                
                # Thermalisation buffer
                j = 0
                while j < tbuffer4:
                    if evo == "Simultaneous":
                        self.sim_step_metro()
                    elif evo == "Alternating":
                        self.alt_step_metro()
                    elif evo == "Systematic":
                        self.sys_step_metro()
                    elif evo == "Randomised":
                        self.ran_step_metro()
                    else:
                        messagebox.showinfo("Error", "Error running simulation")
                    j += 1
                
                # Magnetisation
                k = 0
                M = [] # Array for magnetisation
                
                while k < susstep:
                    if evo == "Simultaneous":
                        self.sim_step_metro()
                    elif evo == "Alternating":
                        self.alt_step_metro()
                    elif evo == "Systematic":
                        self.sys_step_metro()
                    elif evo == "Randomised":
                        self.ran_step_metro()
                    else:
                        messagebox.showinfo("Error", "Error running simulation")
                    
                    data = self.data
                    M_array = np.array(data)
                    
                    if model == "Ising Model":
                        Ein = np.sum(M_array)
                    else:
                        M_ein = complex(np.sum(np.cos(M_array)), np.sum(np.sin(M_array)))
                        Ein = abs(M_ein)
                    
                    M.append(Ein)
                    k += 1
                
                # Determines magnetic susceptibility from standard deviation in magnetisation
                if i == 0:
                    sus = 0
                else:
                    sus = ((np.std(M))**2)/(i)
                
                addon = [i, sus]
                S.append(addon)
                i += istep
            
            self.run = False
        
        end = time.time()
        if self.runtime.get():
            timetaken = end - start
            print("Run time / s: " + str(timetaken))
        
        T = np.array(S)[:,0]
        suscep = np.array(S)[:,1]
        product = np.multiply(T, suscep)
        
        # Pick out 5 largest magnetic susceptibilities and finds their weighted mean
        suscep_sort = np.argsort(suscep)
        largest = np.array(suscep_sort)[-5:]
        n = 0
        suscep_max = []
        product_max = []
        
        while n < 5:
            p = largest[n]
            suscep_max.append(suscep[p])
            product_max.append(product[p])
            n += 1
        
        sum_product = np.sum(product_max)
        sum_suscep = np.sum(suscep_max)
        mean = np.round_(sum_product/sum_suscep, decimals = 3)
        
        # Plot
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.plot(T, suscep, 'k.')
        plt.xlabel("Temperature")
        plt.ylabel("Magnetic susceptibility")
        plt.title("Temperature-dependence of magnetic susceptibility\n" + "Estimated critical temperature = " + str(mean))
        plt.show()

    def spin(self, *args):
        ''' Creates new window for (9) Spin Stiffness '''
        window9 = tk.Toplevel(root)
        window9.title("(9) Spin Stiffness")
        
        window9.temprange12_label = ttk.Label(window9, text = "Lower temperature")
        window9.temprange12_entry = ttk.Entry(window9, textvariable = self.temprange12)
        window9.temprange13_label = ttk.Label(window9, text = "Upper temperature")
        window9.temprange13_entry = ttk.Entry(window9, textvariable = self.temprange13)
        window9.temppoints_label = ttk.Label(window9, text = "No. of sampling points")
        window9.temppoints_entry = ttk.Entry(window9, textvariable = self.samp7)
        window9.tbuffer4_label = ttk.Label(window9, text = "Thermalisation buffer")
        window9.tbuffer4_entry = ttk.Entry(window9, textvariable = self.tbuffer5)
        window9.steps_label = ttk.Label(window9, text = "Number of steps")
        window9.steps_entry = ttk.Entry(window9, textvariable = self.spinstep)
        window9.rep_label = ttk.Label(window9, text = "Repetition")
        window9.rep_entry = ttk.Entry(window9, textvariable = self.spinrep)
        window9.spin_start = ttk.Button(window9, text = "Start", command = self.spinstart)
        
        window9.temprange12_label.grid(column = 0, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.temprange12_entry.grid(column = 1, row = 0, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.temprange13_label.grid(column = 0, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.temprange13_entry.grid(column = 1, row = 1, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.temppoints_label.grid(column = 0, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.temppoints_entry.grid(column = 1, row = 2, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.tbuffer4_label.grid(column = 0, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.tbuffer4_entry.grid(column = 1, row = 3, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.steps_label.grid(column = 0, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.steps_entry.grid(column = 1, row = 4, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.rep_label.grid(column = 0, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.rep_entry.grid(column = 1, row = 5, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
        window9.spin_start.grid(column = 0, row = 6, sticky = (tk.N, tk.W, tk.E, tk.S), padx = 3, pady = 3)
    
    def spinstart(self, *args):
        ''' Runs simulation for a range of temperatures, and plots spin stiffness against temperature '''
        # Get simulation parameters
        self.model.set("XY Model") # This simulation does not apply to the Ising model
        self.update_canvas()
        evo = self.evolution.get()
        temprange12 = self.temprange12.get()
        temprange13 = self.temprange13.get()
        samp7 = self.samp7.get()
        tbuffer5 = self.tbuffer5.get()
        spinstep = self.spinstep.get()
        J = self.J.get()
        N = self.N.get()
        rep = self.spinrep.get()
        start = time.time()
        
        # Run simulation
        self.run = True
        istep = (temprange13 - temprange12)/(samp7 - 1) # Temperature step
        Ups = [] # Holder for main data
        
        while self.run == True:
            i = temprange12
            
            # For each temperature
            while i < temprange13 + istep/2:
                self.temp.set(i)
                m = 0
                ups_cos = []
                ups_sin = []
                
                # For each repetition
                while m < rep:
                    self.set_lattice()
                    self.reset_step()
                    
                    # Thermalisation buffer
                    j = 0
                    while j < tbuffer5:
                        if evo == "Simultaneous":
                            self.sim_step_metro()
                        elif evo == "Alternating":
                            self.alt_step_metro()
                        elif evo == "Systematic":
                            self.sys_step_metro()
                        elif evo == "Randomised":
                            self.ran_step_metro()
                        else:
                            messagebox.showinfo("Error", "Error running simulation")
                        j += 1
                    
                    # Magnetisation
                    k = 0
                    ups = [] # Array for spin stiffness data
                    
                    while k < spinstep:
                        if evo == "Simultaneous":
                            self.sim_step_metro()
                        elif evo == "Alternating":
                            self.alt_step_metro()
                        elif evo == "Systematic":
                            self.sys_step_metro()
                        elif evo == "Randomised":
                            self.ran_step_metro()
                        else:
                            messagebox.showinfo("Error", "Error running simulation")
                        
                        data = np.array(self.data)
                        
                        s_left1 = data[-1]
                        s_left2 = data[:-1]
                        s_left = np.append(s_left1, s_left2) # Array containing spin states of elements directly left of those in original data array
                        cos_data = np.sum(np.cos(data - s_left))
                        sin_data = np.sum(np.sin(data - s_left))
                        ups.append([cos_data, sin_data])
                        k += 1
                    
                    # Determines spin stiffness at a specific temperature
                    cosine = np.mean(np.array(ups)[:,0])
                    sinesq = np.mean(np.square(np.array(ups)[:,1]))
                    stiff = np.absolute((cosine - J*sinesq/i)/(N**2))
                    
                    addon = [i, stiff]
                    Ups.append(addon)
                    
                    m += 1
                
                i += istep
            
            self.run = False
        
        end = time.time()
        if self.runtime.get():
            timetaken = end - start
            print("Run time / s: " + str(timetaken))
        
        T = np.array(Ups)[:,0]
        stiffness = np.array(Ups)[:,1]
        
        # Fit to polynomial
        params = np.polyfit(T, stiffness, 3)
        T_fine = np.linspace(T[0], T[-1], 1001)
        fit = params[0]*(T_fine**3) + params[1]*(T_fine**2) + params[2]*(T_fine**1) + params[3]*np.ones(1001)
        
        # Finds intercept to estimate critical temperature
        line = (2/np.pi)*T_fine
        cross = np.int(np.rint(self.intersect(fit, line, 1)))
        T_crit = np.round_(T_fine[cross], decimals = 3)
        
        # Plot
        plt.rc('text', usetex = True)
        plt.rc('font', family = 'serif')
        plt.plot(T, stiffness, 'k.', label = "Computational data")
        plt.plot(T_fine, fit, 'k-', label = "Fitted curve")
        plt.plot(T_fine, line, 'b-', label = "Intersection line")
        plt.xlabel("Temperature")
        plt.ylabel("Spin Stiffness")
        plt.title("Temperature-dependence of spin stiffness\n" + "Estimated critical temperature = " + str(T_crit))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Phase Transitions")
    
    MainApplication(root).grid(column = 0, row = 0, pady = 0, padx = 0)
    root.mainloop()
