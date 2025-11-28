"""
Advanced Transformer Analysis Tool with Dynamic Simulation
Includes all-day efficiency calculation and real-time ODE simulation
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class TransformerParameters:
    """Transformer parameters for simulation"""
    rated_power: float = 100.0  # kVA
    iron_loss: float = 2.0  # kW
    copper_loss_full: float = 2.0  # kW
    voltage_primary: float = 11.0  # kV
    voltage_secondary: float = 0.415  # kV
    resistance_primary: float = 0.5  # ohms
    resistance_secondary: float = 0.01  # ohms
    reactance_primary: float = 1.0  # ohms
    reactance_secondary: float = 0.02  # ohms
    magnetizing_current: float = 0.02  # per unit
    time_constant: float = 0.1  # seconds


class ODESolver:
    """ODE Solvers for dynamic transformer simulation"""

    @staticmethod
    def euler(f, y0, t_span, dt):
        """Euler method for solving ODEs"""
        t_start, t_end = t_span
        t = np.arange(t_start, t_end, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(1, len(t)):
            y[i] = y[i-1] + dt * f(t[i-1], y[i-1])

        return t, y

    @staticmethod
    def rk45(f, y0, t_span, dt):
        """Runge-Kutta 4th order method (RK45)"""
        t_start, t_end = t_span
        t = np.arange(t_start, t_end, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(1, len(t)):
            k1 = dt * f(t[i-1], y[i-1])
            k2 = dt * f(t[i-1] + dt/2, y[i-1] + k1/2)
            k3 = dt * f(t[i-1] + dt/2, y[i-1] + k2/2)
            k4 = dt * f(t[i-1] + dt, y[i-1] + k3)

            y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

        return t, y


class TransformerSimulator:
    """Transformer dynamic simulation and analysis"""

    def __init__(self, params: TransformerParameters):
        self.params = params
        self.running = False
        self.paused = False

    def calculate_all_day_efficiency(self, full_load_hours, half_load_hours,
                                     no_load_hours):
        """Calculate all-day efficiency of transformer"""
        # Energy output (kWh)
        full_load_output = self.params.rated_power * full_load_hours
        half_load_output = 0.5 * self.params.rated_power * half_load_hours
        total_output = full_load_output + half_load_output

        # Energy losses (kWh)
        total_hours = full_load_hours + half_load_hours + no_load_hours
        iron_loss_total = self.params.iron_loss * total_hours

        copper_loss_full = self.params.copper_loss_full * full_load_hours
        copper_loss_half = self.params.copper_loss_full * (0.5**2) * half_load_hours
        copper_loss_no = 0  # No copper loss at no-load
        copper_loss_total = copper_loss_full + copper_loss_half + copper_loss_no

        total_losses = iron_loss_total + copper_loss_total

        # All-day efficiency
        total_input = total_output + total_losses
        efficiency = (total_output / total_input) * 100 if total_input > 0 else 0

        return {
            'efficiency': efficiency,
            'total_output': total_output,
            'total_input': total_input,
            'iron_losses': iron_loss_total,
            'copper_losses': copper_loss_total,
            'total_losses': total_losses
        }

    def transformer_differential_equation(self, t, y):
        """
        Differential equations for transformer dynamic behavior
        y[0] = flux (Wb)
        y[1] = primary current (A)
        y[2] = secondary current (A)
        y[3] = core temperature (°C)
        """
        flux, i_primary, i_secondary, temp = y

        # Voltage applied (with time-varying load)
        v_applied = self.params.voltage_primary * 1000 * np.sin(2 * np.pi * 50 * t)

        # Load variation (simulating daily load cycle)
        load_factor = 0.5 + 0.5 * np.sin(2 * np.pi * t / 24)

        # Flux derivative (Faraday's law)
        d_flux = (v_applied - self.params.resistance_primary * i_primary) / 100

        # Primary current derivative
        d_i_primary = (v_applied - self.params.resistance_primary * i_primary -
                      self.params.reactance_primary * i_primary) / self.params.time_constant

        # Secondary current derivative (depends on load)
        load_resistance = (self.params.voltage_secondary * 1000) / (self.params.rated_power * 1000 * load_factor)
        d_i_secondary = -(self.params.resistance_secondary + load_resistance) * i_secondary / self.params.time_constant

        # Temperature derivative (thermal dynamics)
        heat_generated = (self.params.iron_loss +
                         self.params.copper_loss_full * (i_primary/100)**2)
        heat_dissipated = 0.1 * (temp - 25)  # Cooling
        d_temp = (heat_generated - heat_dissipated) / 10

        return np.array([d_flux, d_i_primary, d_i_secondary, d_temp])

    def simulate_dynamic(self, method='rk45', duration=10, dt=0.01):
        """Run dynamic simulation"""
        y0 = np.array([0.0, 0.0, 0.0, 25.0])  # Initial conditions
        t_span = (0, duration)

        solver = ODESolver()
        if method == 'rk45':
            t, y = solver.rk45(self.transformer_differential_equation, y0, t_span, dt)
        else:
            t, y = solver.euler(self.transformer_differential_equation, y0, t_span, dt)

        return t, y


def calculate_voltage_regulation(power_kva: float, line_voltage_kv: float,
                                 resistance_ohm: float, reactance_ohm: float,
                                 power_factor: float, leading: bool = False) -> Tuple[float, complex, complex]:
    """Calculate voltage regulation for a three-phase alternator.

    Returns percentage regulation, generated emf (phase), and terminal voltage (phase).
    """
    vt_phase = (line_voltage_kv * 1000) / np.sqrt(3)
    ia = (power_kva * 1000) / (np.sqrt(3) * line_voltage_kv * 1000)

    angle = np.arccos(power_factor)
    if leading:
        angle = -angle

    ia_complex = ia * np.cos(angle) + 1j * ia * np.sin(angle)
    impedance = resistance_ohm + 1j * reactance_ohm

    e_phase = vt_phase + ia_complex * impedance
    regulation = (np.abs(e_phase) - vt_phase) / vt_phase * 100

    return regulation, e_phase, vt_phase


class TransformerGUI:
    """Main GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Transformer Analysis & Simulation Tool")
        self.root.geometry("1400x900")

        # Initialize parameters
        self.params = TransformerParameters()
        self.simulator = TransformerSimulator(self.params)
        self.simulation_thread = None
        self.simulation_running = False

        # Color scheme
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#4CAF50"

        # Configure root
        self.root.configure(bg=self.bg_color)

        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)

        # Create UI
        self.create_menu()
        self.create_main_interface()

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Analysis", command=self.reset_simulation)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Start", command=self.start_simulation)
        sim_menu.add_command(label="Stop", command=self.stop_simulation)
        sim_menu.add_command(label="Reset", command=self.reset_simulation)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)

    def create_main_interface(self):
        """Create main interface with notebook tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create tabs
        self.create_efficiency_tab()
        self.create_dynamic_simulation_tab()
        self.create_load_analysis_tab()
        self.create_thermal_analysis_tab()
        self.create_alternator_tab()
        self.create_comprehensive_tab()

    def create_efficiency_tab(self):
        """Tab for all-day efficiency calculation"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="All-Day Efficiency")

        # Main container with grid layout
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Left panel - Input parameters
        left_frame = ttk.LabelFrame(main_frame, text="Transformer Parameters", padding=10)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Parameter inputs
        params_inputs = [
            ("Rated Power (kVA):", 'rated_power', 100.0),
            ("Iron Loss (kW):", 'iron_loss', 2.0),
            ("Full-Load Copper Loss (kW):", 'copper_loss_full', 2.0),
        ]

        self.efficiency_vars = {}
        for i, (label, key, default) in enumerate(params_inputs):
            ttk.Label(left_frame, text=label).grid(row=i, column=0, sticky='w', pady=5)
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(left_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.efficiency_vars[key] = var

        # Operating hours
        ttk.Label(left_frame, text="\nOperating Schedule (24h):",
                 font=('Arial', 10, 'bold')).grid(row=len(params_inputs),
                                                   column=0, columnspan=2, pady=10)

        schedule_inputs = [
            ("Full-Load Hours:", 'full_load_hours', 4.0),
            ("Half-Load Hours:", 'half_load_hours', 6.0),
            ("No-Load Hours:", 'no_load_hours', 14.0),
        ]

        for i, (label, key, default) in enumerate(schedule_inputs):
            row = len(params_inputs) + 1 + i
            ttk.Label(left_frame, text=label).grid(row=row, column=0, sticky='w', pady=5)
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(left_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, padx=5, pady=5)
            self.efficiency_vars[key] = var

        # Calculate button
        calc_btn = ttk.Button(left_frame, text="Calculate Efficiency",
                             command=self.calculate_efficiency)
        calc_btn.grid(row=len(params_inputs)+len(schedule_inputs)+1,
                     column=0, columnspan=2, pady=20)

        # Right panel - Results
        right_frame = ttk.LabelFrame(main_frame, text="Calculation Results", padding=10)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        self.results_text = scrolledtext.ScrolledText(right_frame, width=50, height=20,
                                                      font=('Courier', 10))
        self.results_text.pack(fill='both', expand=True)

        # Bottom panel - Visualization
        bottom_frame = ttk.LabelFrame(main_frame, text="Energy Distribution", padding=10)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

        self.efficiency_figure = Figure(figsize=(12, 4), dpi=100)
        self.efficiency_canvas = FigureCanvasTkAgg(self.efficiency_figure, bottom_frame)
        self.efficiency_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=2)
        main_frame.rowconfigure(1, weight=1)

    def create_dynamic_simulation_tab(self):
        """Tab for dynamic simulation with ODE solvers"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Dynamic Simulation")

        # Control panel
        control_frame = ttk.LabelFrame(tab, text="Simulation Controls", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)

        # Solver selection
        ttk.Label(control_frame, text="ODE Solver:").grid(row=0, column=0, padx=5)
        self.solver_var = tk.StringVar(value='rk45')
        solver_combo = ttk.Combobox(control_frame, textvariable=self.solver_var,
                                    values=['rk45', 'euler'], state='readonly', width=10)
        solver_combo.grid(row=0, column=1, padx=5)

        # Duration
        ttk.Label(control_frame, text="Duration (s):").grid(row=0, column=2, padx=5)
        self.duration_var = tk.DoubleVar(value=10.0)
        duration_entry = ttk.Entry(control_frame, textvariable=self.duration_var, width=10)
        duration_entry.grid(row=0, column=3, padx=5)

        # Time step
        ttk.Label(control_frame, text="Time Step (s):").grid(row=0, column=4, padx=5)
        self.dt_var = tk.DoubleVar(value=0.01)
        dt_entry = ttk.Entry(control_frame, textvariable=self.dt_var, width=10)
        dt_entry.grid(row=0, column=5, padx=5)

        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=0, column=6, padx=20)

        self.start_btn = ttk.Button(btn_frame, text="▶ Start",
                                    command=self.start_simulation, width=10)
        self.start_btn.pack(side='left', padx=2)

        self.stop_btn = ttk.Button(btn_frame, text="⬛ Stop",
                                   command=self.stop_simulation, width=10, state='disabled')
        self.stop_btn.pack(side='left', padx=2)

        self.reset_btn = ttk.Button(btn_frame, text="↻ Reset",
                                    command=self.reset_simulation, width=10)
        self.reset_btn.pack(side='left', padx=2)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                           maximum=100, length=200)
        self.progress_bar.grid(row=1, column=0, columnspan=7, pady=10, sticky='ew')

        # Visualization frame
        viz_frame = ttk.LabelFrame(tab, text="Real-Time Simulation Results", padding=10)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.sim_figure = Figure(figsize=(14, 8), dpi=100)
        self.sim_canvas = FigureCanvasTkAgg(self.sim_figure, viz_frame)
        self.sim_canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_load_analysis_tab(self):
        """Tab for load analysis with sliders"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Load Analysis")

        # Control frame with sliders
        control_frame = ttk.LabelFrame(tab, text="Load Control", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)

        # Load sliders
        self.load_sliders = {}
        slider_params = [
            ("Load Factor (%)", 'load_factor', 0, 120, 100),
            ("Power Factor", 'power_factor', 0, 100, 85),
            ("Frequency (Hz)", 'frequency', 45, 65, 50),
        ]

        for i, (label, key, min_val, max_val, default) in enumerate(slider_params):
            frame = ttk.Frame(control_frame)
            frame.grid(row=i, column=0, columnspan=3, sticky='ew', pady=5)

            ttk.Label(frame, text=label, width=20).pack(side='left')

            var = tk.DoubleVar(value=default)
            slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var,
                              orient='horizontal', length=400,
                              command=lambda v, k=key: self.update_load_analysis())
            slider.pack(side='left', padx=10, fill='x', expand=True)

            value_label = ttk.Label(frame, text=f"{default:.1f}", width=10)
            value_label.pack(side='left')

            self.load_sliders[key] = (var, value_label, slider)

        # Update button
        update_btn = ttk.Button(control_frame, text="Update Analysis",
                               command=self.update_load_analysis)
        update_btn.grid(row=len(slider_params), column=0, columnspan=3, pady=10)

        # Results visualization
        viz_frame = ttk.LabelFrame(tab, text="Load Analysis Results", padding=10)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.load_figure = Figure(figsize=(14, 8), dpi=100)
        self.load_canvas = FigureCanvasTkAgg(self.load_figure, viz_frame)
        self.load_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Initial plot
        self.update_load_analysis()

    def create_alternator_tab(self):
        """Tab for alternator voltage regulation and phasor diagrams"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Alternator Lab")

        container = ttk.Frame(tab)
        container.pack(fill='both', expand=True, padx=10, pady=10)

        # Input parameters
        input_frame = ttk.LabelFrame(container, text="Machine Parameters", padding=10)
        input_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        defaults = {
            'power_kva': 500.0,
            'voltage_kv': 1.1,
            'frequency': 50.0,
            'resistance': 0.2,
            'reactance': 1.5
        }

        self.alt_vars = {}
        labels = [
            ("Rated Power (kVA)", 'power_kva'),
            ("Line Voltage (kV)", 'voltage_kv'),
            ("Frequency (Hz)", 'frequency'),
            ("Armature Resistance (Ω/phase)", 'resistance'),
            ("Synchronous Reactance (Ω/phase)", 'reactance'),
        ]

        for i, (label, key) in enumerate(labels):
            ttk.Label(input_frame, text=label).grid(row=i, column=0, sticky='w', pady=3)
            var = tk.DoubleVar(value=defaults[key])
            entry = ttk.Entry(input_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, padx=5, pady=3)
            self.alt_vars[key] = var

        # Power factor controls
        ttk.Label(input_frame, text="Power Factor").grid(row=len(labels), column=0, sticky='w', pady=3)
        self.alt_pf_var = tk.DoubleVar(value=0.8)
        pf_slider = ttk.Scale(input_frame, from_=0.1, to=1.0, variable=self.alt_pf_var,
                              orient='horizontal', length=180,
                              command=lambda v: self.alt_pf_label.config(text=f"{float(v):.3f}"))
        pf_slider.grid(row=len(labels), column=1, padx=5, pady=3, sticky='ew')
        self.alt_pf_label = ttk.Label(input_frame, text="0.800")
        self.alt_pf_label.grid(row=len(labels), column=2, padx=5)

        ttk.Label(input_frame, text="Power Factor Mode").grid(row=len(labels)+1, column=0, sticky='w', pady=3)
        self.alt_pf_mode = tk.StringVar(value='lagging')
        pf_combo = ttk.Combobox(input_frame, textvariable=self.alt_pf_mode, values=['lagging', 'leading'],
                                state='readonly', width=10)
        pf_combo.grid(row=len(labels)+1, column=1, padx=5, pady=3, sticky='w')

        ttk.Label(input_frame, text="Load Level (%)").grid(row=len(labels)+2, column=0, sticky='w', pady=3)
        self.alt_load_var = tk.DoubleVar(value=100.0)
        load_slider = ttk.Scale(input_frame, from_=20, to=120, variable=self.alt_load_var,
                                orient='horizontal', length=180,
                                command=lambda v: self.alt_load_label.config(text=f"{float(v):.1f}%"))
        load_slider.grid(row=len(labels)+2, column=1, padx=5, pady=3, sticky='ew')
        self.alt_load_label = ttk.Label(input_frame, text="100.0%")
        self.alt_load_label.grid(row=len(labels)+2, column=2, padx=5)

        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=len(labels)+3, column=0, columnspan=3, pady=10)

        ttk.Button(btn_frame, text="Compute Regulation", command=self.compute_alternator_regulation,
                   width=20).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Solve 0.8 pf Cases", command=self.solve_reference_cases,
                   width=20).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset_alternator_inputs,
                   width=12).pack(side='left', padx=5)

        # Results and visualization
        result_frame = ttk.LabelFrame(container, text="Results & Phasor Diagram", padding=10)
        result_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        self.alt_results = scrolledtext.ScrolledText(result_frame, width=60, height=18, font=('Courier', 10))
        self.alt_results.pack(fill='x', expand=False, pady=5)

        self.alt_figure = Figure(figsize=(8, 6), dpi=100)
        self.alt_canvas = FigureCanvasTkAgg(self.alt_figure, result_frame)
        self.alt_canvas.get_tk_widget().pack(fill='both', expand=True)

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=1)

    def create_comprehensive_tab(self):
        """Advanced dashboard combining machine insights"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Comprehensive Analysis")

        control = ttk.LabelFrame(tab, text="Scenario Controls", padding=10)
        control.pack(fill='x', padx=10, pady=5)

        self.comp_load_var = tk.DoubleVar(value=80)
        self.comp_pf_var = tk.DoubleVar(value=0.95)
        self.comp_solver = tk.StringVar(value='rk45')

        ttk.Label(control, text="Load Swing (%)").grid(row=0, column=0, padx=5)
        ttk.Scale(control, from_=10, to=120, variable=self.comp_load_var, orient='horizontal', length=200,
                  command=lambda v: self.comp_load_label.config(text=f"{float(v):.1f}%"))\
            .grid(row=0, column=1, padx=5, sticky='ew')
        self.comp_load_label = ttk.Label(control, text="80.0%")
        self.comp_load_label.grid(row=0, column=2, padx=5)

        ttk.Label(control, text="Target Power Factor").grid(row=0, column=3, padx=5)
        ttk.Scale(control, from_=0.6, to=1.0, variable=self.comp_pf_var, orient='horizontal', length=200,
                  command=lambda v: self.comp_pf_label.config(text=f"{float(v):.3f}"))\
            .grid(row=0, column=4, padx=5, sticky='ew')
        self.comp_pf_label = ttk.Label(control, text="0.950")
        self.comp_pf_label.grid(row=0, column=5, padx=5)

        ttk.Label(control, text="ODE Solver").grid(row=0, column=6, padx=5)
        ttk.Combobox(control, textvariable=self.comp_solver, values=['rk45', 'euler'], state='readonly', width=10)\
            .grid(row=0, column=7, padx=5)

        ttk.Button(control, text="Run Comprehensive Study", command=self.update_comprehensive_dashboard,
                   width=28).grid(row=0, column=8, padx=10)

        viz = ttk.LabelFrame(tab, text="Dashboard", padding=10)
        viz.pack(fill='both', expand=True, padx=10, pady=5)

        self.comp_figure = Figure(figsize=(14, 8), dpi=100)
        self.comp_canvas = FigureCanvasTkAgg(self.comp_figure, viz)
        self.comp_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.update_comprehensive_dashboard()

    def create_thermal_analysis_tab(self):
        """Tab for thermal analysis"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Thermal Analysis")

        # Parameters frame
        param_frame = ttk.LabelFrame(tab, text="Thermal Parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        thermal_params = [
            ("Ambient Temperature (°C):", 'ambient_temp', 25.0),
            ("Cooling Factor:", 'cooling_factor', 0.1),
            ("Thermal Time Constant (min):", 'thermal_tau', 60.0),
        ]

        self.thermal_vars = {}
        for i, (label, key, default) in enumerate(thermal_params):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(param_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.thermal_vars[key] = var

        # Simulate thermal button
        sim_thermal_btn = ttk.Button(param_frame, text="Simulate Thermal Behavior",
                                     command=self.simulate_thermal)
        sim_thermal_btn.grid(row=len(thermal_params), column=0, columnspan=2, pady=10)

        # Visualization
        viz_frame = ttk.LabelFrame(tab, text="Thermal Response", padding=10)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.thermal_figure = Figure(figsize=(14, 8), dpi=100)
        self.thermal_canvas = FigureCanvasTkAgg(self.thermal_figure, viz_frame)
        self.thermal_canvas.get_tk_widget().pack(fill='both', expand=True)

    def calculate_efficiency(self):
        """Calculate and display all-day efficiency"""
        try:
            # Update parameters
            self.params.rated_power = self.efficiency_vars['rated_power'].get()
            self.params.iron_loss = self.efficiency_vars['iron_loss'].get()
            self.params.copper_loss_full = self.efficiency_vars['copper_loss_full'].get()

            # Get operating hours
            full_load_hours = self.efficiency_vars['full_load_hours'].get()
            half_load_hours = self.efficiency_vars['half_load_hours'].get()
            no_load_hours = self.efficiency_vars['no_load_hours'].get()

            # Validate total hours
            if abs(full_load_hours + half_load_hours + no_load_hours - 24) > 0.01:
                messagebox.showwarning("Invalid Input",
                                      "Total operating hours must equal 24 hours!")
                return

            # Calculate efficiency
            results = self.simulator.calculate_all_day_efficiency(
                full_load_hours, half_load_hours, no_load_hours
            )

            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "="*60 + "\n")
            self.results_text.insert(tk.END, "  TRANSFORMER ALL-DAY EFFICIENCY ANALYSIS\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")

            self.results_text.insert(tk.END, "INPUT PARAMETERS:\n")
            self.results_text.insert(tk.END, f"  Rated Power: {self.params.rated_power} kVA\n")
            self.results_text.insert(tk.END, f"  Iron Loss: {self.params.iron_loss} kW\n")
            self.results_text.insert(tk.END, f"  Full-Load Copper Loss: {self.params.copper_loss_full} kW\n\n")

            self.results_text.insert(tk.END, "OPERATING SCHEDULE:\n")
            self.results_text.insert(tk.END, f"  Full-Load: {full_load_hours} hours\n")
            self.results_text.insert(tk.END, f"  Half-Load: {half_load_hours} hours\n")
            self.results_text.insert(tk.END, f"  No-Load: {no_load_hours} hours\n\n")

            self.results_text.insert(tk.END, "-"*60 + "\n")
            self.results_text.insert(tk.END, "ENERGY CALCULATION:\n")
            self.results_text.insert(tk.END, f"  Total Output Energy: {results['total_output']:.2f} kWh\n")
            self.results_text.insert(tk.END, f"  Iron Losses: {results['iron_losses']:.2f} kWh\n")
            self.results_text.insert(tk.END, f"  Copper Losses: {results['copper_losses']:.2f} kWh\n")
            self.results_text.insert(tk.END, f"  Total Losses: {results['total_losses']:.2f} kWh\n")
            self.results_text.insert(tk.END, f"  Total Input Energy: {results['total_input']:.2f} kWh\n\n")

            self.results_text.insert(tk.END, "="*60 + "\n")
            self.results_text.insert(tk.END, f"  ALL-DAY EFFICIENCY: {results['efficiency']:.4f}%\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")

            # Additional analysis
            loss_percentage = (results['total_losses'] / results['total_input']) * 100
            self.results_text.insert(tk.END, f"Loss Percentage: {loss_percentage:.4f}%\n")
            self.results_text.insert(tk.END, f"Iron Loss Contribution: {(results['iron_losses']/results['total_losses'])*100:.2f}%\n")
            self.results_text.insert(tk.END, f"Copper Loss Contribution: {(results['copper_losses']/results['total_losses'])*100:.2f}%\n")

            # Plot results
            self.plot_efficiency_results(results, full_load_hours, half_load_hours, no_load_hours)

        except Exception as e:
            messagebox.showerror("Calculation Error", f"Error: {str(e)}")

    def plot_efficiency_results(self, results, full_hours, half_hours, no_hours):
        """Plot efficiency analysis results"""
        self.efficiency_figure.clear()

        # Create subplots
        gs = self.efficiency_figure.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Energy distribution pie chart
        ax1 = self.efficiency_figure.add_subplot(gs[0, 0])
        energy_data = [results['total_output'], results['total_losses']]
        colors = ['#4CAF50', '#F44336']
        ax1.pie(energy_data, labels=['Output Energy', 'Total Losses'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Energy Distribution')

        # 2. Loss breakdown pie chart
        ax2 = self.efficiency_figure.add_subplot(gs[0, 1])
        loss_data = [results['iron_losses'], results['copper_losses']]
        colors2 = ['#FF9800', '#2196F3']
        ax2.pie(loss_data, labels=['Iron Losses', 'Copper Losses'],
               autopct='%1.1f%%', colors=colors2, startangle=90)
        ax2.set_title('Loss Breakdown')

        # 3. Operating schedule
        ax3 = self.efficiency_figure.add_subplot(gs[0, 2])
        schedule_data = [full_hours, half_hours, no_hours]
        labels = ['Full-Load', 'Half-Load', 'No-Load']
        colors3 = ['#F44336', '#FF9800', '#4CAF50']
        bars = ax3.bar(labels, schedule_data, color=colors3)
        ax3.set_ylabel('Hours')
        ax3.set_title('Operating Schedule (24h)')
        ax3.set_ylim(0, 24)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}h', ha='center', va='bottom')

        # 4. Load profile over 24 hours
        ax4 = self.efficiency_figure.add_subplot(gs[1, :])
        hours = np.array([0, full_hours, full_hours, full_hours+half_hours,
                         full_hours+half_hours, 24])
        load = np.array([100, 100, 50, 50, 0, 0])
        ax4.plot(hours, load, 'b-', linewidth=2, marker='o')
        ax4.fill_between(hours, load, alpha=0.3)
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Load (%)')
        ax4.set_title('24-Hour Load Profile')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 24)
        ax4.set_ylim(0, 120)

        self.efficiency_canvas.draw()

    def start_simulation(self):
        """Start dynamic simulation"""
        if self.simulation_running:
            messagebox.showinfo("Info", "Simulation already running!")
            return

        self.simulation_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_var.set(0)

        # Run simulation in separate thread
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def run_simulation(self):
        """Run the simulation (called in thread)"""
        try:
            method = self.solver_var.get()
            duration = self.duration_var.get()
            dt = self.dt_var.get()

            # Update progress
            self.root.after(0, lambda: self.progress_var.set(10))

            # Run simulation
            t, y = self.simulator.simulate_dynamic(method, duration, dt)

            # Update progress
            self.root.after(0, lambda: self.progress_var.set(50))

            # Plot results
            self.root.after(0, lambda: self.plot_simulation_results(t, y))

            # Complete
            self.root.after(0, lambda: self.progress_var.set(100))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", str(e)))
        finally:
            self.simulation_running = False
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))

    def plot_simulation_results(self, t, y):
        """Plot dynamic simulation results"""
        self.sim_figure.clear()

        # Create subplots
        ax1 = self.sim_figure.add_subplot(2, 2, 1)
        ax1.plot(t, y[:, 0], 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Flux (Wb)')
        ax1.set_title('Magnetic Flux')
        ax1.grid(True, alpha=0.3)

        ax2 = self.sim_figure.add_subplot(2, 2, 2)
        ax2.plot(t, y[:, 1], 'r-', linewidth=2, label='Primary')
        ax2.plot(t, y[:, 2], 'g-', linewidth=2, label='Secondary')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('Winding Currents')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = self.sim_figure.add_subplot(2, 2, 3)
        ax3.plot(t, y[:, 3], 'orange', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('Core Temperature')
        ax3.grid(True, alpha=0.3)

        ax4 = self.sim_figure.add_subplot(2, 2, 4)
        # Calculate instantaneous power losses
        power_loss = self.params.iron_loss + self.params.copper_loss_full * (y[:, 1]/100)**2
        ax4.plot(t, power_loss, 'purple', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Power Loss (kW)')
        ax4.set_title('Instantaneous Power Losses')
        ax4.grid(True, alpha=0.3)

        self.sim_figure.tight_layout()
        self.sim_canvas.draw()

    def stop_simulation(self):
        """Stop the simulation"""
        self.simulation_running = False
        self.stop_btn.config(state='disabled')
        self.start_btn.config(state='normal')

    def reset_simulation(self):
        """Reset simulation"""
        self.stop_simulation()
        self.progress_var.set(0)
        self.sim_figure.clear()
        self.sim_canvas.draw()

    def update_load_analysis(self, *args):
        """Update load analysis plots"""
        # Update slider value labels
        for key, (var, label, slider) in self.load_sliders.items():
            label.config(text=f"{var.get():.1f}")

        # Get current values
        load_factor = self.load_sliders['load_factor'][0].get() / 100
        power_factor = self.load_sliders['power_factor'][0].get() / 100
        frequency = self.load_sliders['frequency'][0].get()

        # Calculate various parameters
        apparent_power = self.params.rated_power * load_factor
        real_power = apparent_power * power_factor
        reactive_power = apparent_power * np.sqrt(1 - power_factor**2)

        # Calculate losses
        copper_loss = self.params.copper_loss_full * (load_factor**2)
        iron_loss = self.params.iron_loss
        total_loss = copper_loss + iron_loss

        # Calculate efficiency
        efficiency = (real_power / (real_power + total_loss)) * 100 if (real_power + total_loss) > 0 else 0

        # Voltage regulation
        voltage_drop = load_factor * 0.05  # Simplified model

        # Plot results
        self.load_figure.clear()

        gs = self.load_figure.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Power triangle
        ax1 = self.load_figure.add_subplot(gs[0, 0])
        ax1.arrow(0, 0, real_power, 0, head_width=2, head_length=2, fc='blue', ec='blue')
        ax1.arrow(real_power, 0, 0, reactive_power, head_width=2, head_length=2, fc='red', ec='red')
        ax1.plot([0, real_power], [0, reactive_power], 'g--', linewidth=2)
        ax1.text(real_power/2, -5, f'P = {real_power:.2f} kW', ha='center')
        ax1.text(real_power+5, reactive_power/2, f'Q = {reactive_power:.2f} kVAR', ha='left')
        ax1.text(real_power/2, reactive_power/2+5, f'S = {apparent_power:.2f} kVA', ha='center')
        ax1.set_xlim(-5, max(100, real_power+10))
        ax1.set_ylim(-10, max(50, reactive_power+10))
        ax1.set_xlabel('Real Power (kW)')
        ax1.set_ylabel('Reactive Power (kVAR)')
        ax1.set_title('Power Triangle')
        ax1.grid(True, alpha=0.3)

        # 2. Loss distribution
        ax2 = self.load_figure.add_subplot(gs[0, 1])
        losses = [copper_loss, iron_loss]
        colors = ['#2196F3', '#FF9800']
        wedges, texts, autotexts = ax2.pie(losses, labels=['Copper Loss', 'Iron Loss'],
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title(f'Total Losses: {total_loss:.2f} kW')

        # 3. Efficiency vs Load
        ax3 = self.load_figure.add_subplot(gs[0, 2])
        load_range = np.linspace(0.1, 1.2, 50)
        eff_range = []
        for lf in load_range:
            p = self.params.rated_power * lf * power_factor
            cu_loss = self.params.copper_loss_full * (lf**2)
            total_l = cu_loss + iron_loss
            eff = (p / (p + total_l)) * 100 if (p + total_l) > 0 else 0
            eff_range.append(eff)

        ax3.plot(load_range * 100, eff_range, 'b-', linewidth=2)
        ax3.axvline(load_factor * 100, color='r', linestyle='--', label='Current Load')
        ax3.set_xlabel('Load (%)')
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('Efficiency vs Load')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 120)

        # 4. Voltage regulation
        ax4 = self.load_figure.add_subplot(gs[1, 0])
        load_points = np.linspace(0, 1.2, 50)
        v_secondary = []
        for lf in load_points:
            vdrop = lf * 0.05
            v_secondary.append((1 - vdrop) * 100)

        ax4.plot(load_points * 100, v_secondary, 'g-', linewidth=2)
        ax4.axhline(100, color='k', linestyle='--', alpha=0.5, label='Rated Voltage')
        ax4.axvline(load_factor * 100, color='r', linestyle='--', label='Current Load')
        ax4.set_xlabel('Load (%)')
        ax4.set_ylabel('Secondary Voltage (%)')
        ax4.set_title('Voltage Regulation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 120)

        # 5. Current waveform (simplified)
        ax5 = self.load_figure.add_subplot(gs[1, 1])
        t_wave = np.linspace(0, 3/frequency, 1000)
        i_wave = load_factor * 100 * np.sin(2 * np.pi * frequency * t_wave)
        ax5.plot(t_wave * 1000, i_wave, 'r-', linewidth=2)
        ax5.set_xlabel('Time (ms)')
        ax5.set_ylabel('Current (A)')
        ax5.set_title(f'Current Waveform ({frequency} Hz)')
        ax5.grid(True, alpha=0.3)

        # 6. Summary table
        ax6 = self.load_figure.add_subplot(gs[1, 2])
        ax6.axis('off')

        summary_data = [
            ['Parameter', 'Value'],
            ['Load Factor', f'{load_factor*100:.1f}%'],
            ['Power Factor', f'{power_factor:.3f}'],
            ['Apparent Power', f'{apparent_power:.2f} kVA'],
            ['Real Power', f'{real_power:.2f} kW'],
            ['Reactive Power', f'{reactive_power:.2f} kVAR'],
            ['Total Losses', f'{total_loss:.2f} kW'],
            ['Efficiency', f'{efficiency:.2f}%'],
            ['Voltage Drop', f'{voltage_drop*100:.2f}%'],
        ]

        table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax6.set_title('Operating Parameters', pad=20, fontweight='bold')

        self.load_canvas.draw()

    def compute_alternator_regulation(self):
        """Compute voltage regulation and plot phasor diagram"""
        try:
            power_kva = self.alt_vars['power_kva'].get() * (self.alt_load_var.get() / 100)
            voltage_kv = self.alt_vars['voltage_kv'].get()
            resistance = self.alt_vars['resistance'].get()
            reactance = self.alt_vars['reactance'].get()
            pf = max(0.1, min(1.0, self.alt_pf_var.get()))
            leading = self.alt_pf_mode.get() == 'leading'

            reg, e_phase, vt_phase = calculate_voltage_regulation(power_kva, voltage_kv,
                                                                  resistance, reactance,
                                                                  pf, leading)

            ia = (power_kva * 1000) / (np.sqrt(3) * voltage_kv * 1000)
            angle = np.arccos(pf)
            if leading:
                angle = -angle
            ia_complex = ia * np.cos(angle) + 1j * ia * np.sin(angle)

            self.alt_results.delete(1.0, tk.END)
            self.alt_results.insert(tk.END, "ALTERNATOR VOLTAGE REGULATION\n")
            self.alt_results.insert(tk.END, "="*60 + "\n")
            self.alt_results.insert(tk.END, f"Load: {power_kva:.2f} kVA ({self.alt_load_var.get():.1f}%)\n")
            self.alt_results.insert(tk.END, f"Line Voltage: {voltage_kv:.3f} kV\n")
            self.alt_results.insert(tk.END, f"Armature Resistance: {resistance:.3f} Ω\n")
            self.alt_results.insert(tk.END, f"Synchronous Reactance: {reactance:.3f} Ω\n")
            self.alt_results.insert(tk.END, f"Power Factor: {pf:.3f} ({'leading' if leading else 'lagging'})\n")
            self.alt_results.insert(tk.END, f"Per-phase Terminal Voltage: {vt_phase:.2f} V\n")
            self.alt_results.insert(tk.END, f"Line Current: {ia:.2f} A\n")
            self.alt_results.insert(tk.END, f"Internal Generated EMF: {np.abs(e_phase):.2f} V (phase)\n")
            self.alt_results.insert(tk.END, f"Voltage Regulation: {reg:.3f}%\n")

            self.plot_phasor_diagram(vt_phase, ia_complex, resistance, reactance, e_phase)
        except Exception as e:
            messagebox.showerror("Alternator Calculation Error", str(e))

    def plot_phasor_diagram(self, vt_phase: float, ia: complex, r: float, x: float, emf: complex):
        """Plot phasor diagram for alternator"""
        self.alt_figure.clear()
        ax = self.alt_figure.add_subplot(1, 1, 1)

        vt_vec = vt_phase + 0j
        ia_vec = ia
        ir_drop = ia * r
        ix_drop = ia * 1j * x

        def draw_vector(start: complex, vec: complex, color: str, label: str):
            ax.arrow(start.real, start.imag, vec.real, vec.imag, head_width=0.05*vt_phase,
                     head_length=0.05*vt_phase, fc=color, ec=color, length_includes_head=True)
            ax.text(start.real + vec.real, start.imag + vec.imag, label, color=color, fontsize=9)

        draw_vector(0+0j, vt_vec, 'blue', 'Vt')
        draw_vector(0+0j, ia_vec*vt_phase/abs(vt_phase), 'green', 'Ia (scaled)')
        draw_vector(vt_vec, ir_drop, 'orange', 'IaR')
        draw_vector(vt_vec + ir_drop, ix_drop, 'red', 'IaX')
        draw_vector(0+0j, emf, 'purple', 'E')

        max_mag = max(np.abs([vt_vec, ia_vec*vt_phase/abs(vt_phase), ir_drop, ix_drop, emf])) * 1.3
        ax.set_xlim(-max_mag, max_mag)
        ax.set_ylim(-max_mag, max_mag)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Real Axis (V)')
        ax.set_ylabel('Imag Axis (V)')
        ax.set_title('Alternator Phasor Diagram')
        self.alt_figure.tight_layout()
        self.alt_canvas.draw()

    def solve_reference_cases(self):
        """Solve 0.8 lag/lead regulation for reference problem"""
        try:
            # Set defaults
            self.alt_vars['power_kva'].set(500.0)
            self.alt_vars['voltage_kv'].set(1.1)
            self.alt_vars['resistance'].set(0.2)
            self.alt_vars['reactance'].set(1.5)
            self.alt_pf_var.set(0.8)
            self.alt_load_var.set(100.0)
            self.alt_pf_label.config(text="0.800")
            self.alt_load_label.config(text="100.0%")

            lag_reg, lag_e, vt = calculate_voltage_regulation(500.0, 1.1, 0.2, 1.5, 0.8, leading=False)
            lead_reg, lead_e, _ = calculate_voltage_regulation(500.0, 1.1, 0.2, 1.5, 0.8, leading=True)

            self.alt_results.delete(1.0, tk.END)
            self.alt_results.insert(tk.END, "REFERENCE CASE (500 kVA, 1.1 kV, 50 Hz)\n")
            self.alt_results.insert(tk.END, "="*60 + "\n")
            self.alt_results.insert(tk.END, f"Voltage Regulation @ 0.8 lagging: {lag_reg:.3f}%\n")
            self.alt_results.insert(tk.END, f"Voltage Regulation @ 0.8 leading: {lead_reg:.3f}%\n")
            self.alt_results.insert(tk.END, "Computed using E = V + I(R + jX) per phase." )

            # Plot lagging by default
            ia = (500000) / (np.sqrt(3) * 1100)
            angle = np.arccos(0.8)
            ia_vec = ia * np.cos(angle) - 1j * ia * np.sin(angle)
            self.plot_phasor_diagram(vt, ia_vec, 0.2, 1.5, lag_e)
        except Exception as e:
            messagebox.showerror("Reference Case Error", str(e))

    def reset_alternator_inputs(self):
        """Reset alternator inputs to defaults"""
        defaults = {
            'power_kva': 500.0,
            'voltage_kv': 1.1,
            'frequency': 50.0,
            'resistance': 0.2,
            'reactance': 1.5
        }
        for key, value in defaults.items():
            self.alt_vars[key].set(value)
        self.alt_pf_var.set(0.8)
        self.alt_pf_label.config(text="0.800")
        self.alt_pf_mode.set('lagging')
        self.alt_load_var.set(100.0)
        self.alt_load_label.config(text="100.0%")
        self.alt_results.delete(1.0, tk.END)
        self.alt_figure.clear()
        self.alt_canvas.draw()

    def update_comprehensive_dashboard(self):
        """Create an at-a-glance dashboard for system performance"""
        load = self.comp_load_var.get() / 100
        pf = self.comp_pf_var.get()
        solver = self.comp_solver.get()

        # Quick dynamic run for temperature and current trends
        t, y = self.simulator.simulate_dynamic(method=solver, duration=5, dt=0.02)
        flux, i_prim, i_sec, temp = y.T

        real_power = self.params.rated_power * load * pf
        reactive_power = self.params.rated_power * load * np.sqrt(max(0, 1 - pf**2))
        copper_loss = self.params.copper_loss_full * load**2
        iron_loss = self.params.iron_loss

        self.comp_figure.clear()
        gs = self.comp_figure.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

        ax1 = self.comp_figure.add_subplot(gs[0, 0])
        ax1.plot(t, i_prim, label='Primary', color='tab:red')
        ax1.plot(t, i_sec, label='Secondary', color='tab:blue')
        ax1.set_title('Dynamic Currents')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Current (A)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = self.comp_figure.add_subplot(gs[0, 1])
        ax2.plot(t, flux, color='tab:green')
        ax2.set_title('Flux Linkage Response')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Flux (Wb)')
        ax2.grid(True, alpha=0.3)

        ax3 = self.comp_figure.add_subplot(gs[0, 2])
        efficiency = real_power / (real_power + copper_loss + iron_loss) * 100 if real_power > 0 else 0
        ax3.bar(['Real Power', 'Reactive Power', 'Efficiency'], [real_power, reactive_power, efficiency],
                color=['#4CAF50', '#2196F3', '#FFC107'])
        ax3.set_title('Power Snapshot')
        ax3.set_ylim(0, max(100, real_power + 20))

        ax4 = self.comp_figure.add_subplot(gs[1, 0])
        ax4.plot(t, temp, color='orange')
        ax4.set_title('Core Temperature (5s preview)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Temperature (°C)')
        ax4.grid(True, alpha=0.3)

        ax5 = self.comp_figure.add_subplot(gs[1, 1])
        pf_range = np.linspace(0.6, 1.0, 25)
        reg_curve = []
        for pf_val in pf_range:
            reg, _, _ = calculate_voltage_regulation(self.params.rated_power, self.params.voltage_primary,
                                                     self.params.resistance_primary, self.params.reactance_primary,
                                                     pf_val, leading=False)
            reg_curve.append(reg)
        ax5.plot(pf_range, reg_curve, 'm-')
        ax5.axvline(pf, color='k', linestyle='--', label='Current PF')
        ax5.set_title('Voltage Regulation vs PF (lagging)')
        ax5.set_xlabel('Power Factor')
        ax5.set_ylabel('Regulation (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        ax6 = self.comp_figure.add_subplot(gs[1, 2])
        summary = [
            ['Metric', 'Value'],
            ['Load', f'{self.comp_load_var.get():.1f}%'],
            ['PF Target', f'{pf:.3f}'],
            ['ODE Solver', solver.upper()],
            ['Real Power', f'{real_power:.2f} kW'],
            ['Reactive Power', f'{reactive_power:.2f} kVAR'],
            ['Copper Loss', f'{copper_loss:.2f} kW'],
            ['Iron Loss', f'{iron_loss:.2f} kW'],
            ['Instant Eff.', f'{efficiency:.2f}%']
        ]
        table = ax6.table(cellText=summary, colWidths=[0.55, 0.45], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(color='white', weight='bold')
        ax6.axis('off')
        ax6.set_title('Scenario Summary', pad=10)

        self.comp_figure.tight_layout()
        self.comp_canvas.draw()

    def simulate_thermal(self):
        """Simulate thermal behavior"""
        try:
            ambient_temp = self.thermal_vars['ambient_temp'].get()
            cooling_factor = self.thermal_vars['cooling_factor'].get()
            thermal_tau = self.thermal_vars['thermal_tau'].get() * 60  # Convert to seconds

            # Thermal differential equation
            def thermal_ode(t, T):
                # Heat generated (varying with load cycle)
                load_cycle = 0.5 + 0.5 * np.sin(2 * np.pi * t / 3600)  # 1-hour cycle
                heat_gen = (self.params.iron_loss +
                           self.params.copper_loss_full * load_cycle**2)

                # Heat dissipated
                heat_diss = cooling_factor * (T[0] - ambient_temp)

                # Temperature rate of change
                dT_dt = (heat_gen - heat_diss) / (thermal_tau / 60)

                return np.array([dT_dt])

            # Simulate for 8 hours
            duration = 8 * 3600  # seconds
            dt = 60  # 1 minute steps

            solver = ODESolver()
            t, T = solver.rk45(thermal_ode, np.array([ambient_temp]), (0, duration), dt)

            # Convert time to hours
            t_hours = t / 3600

            # Calculate load profile
            load_profile = 0.5 + 0.5 * np.sin(2 * np.pi * t / 3600)

            # Plot results
            self.thermal_figure.clear()

            ax1 = self.thermal_figure.add_subplot(2, 1, 1)
            ax1.plot(t_hours, T[:, 0], 'r-', linewidth=2)
            ax1.axhline(ambient_temp, color='b', linestyle='--', label='Ambient Temperature')
            ax1.axhline(ambient_temp + 60, color='orange', linestyle='--', label='Warning Level')
            ax1.axhline(ambient_temp + 80, color='red', linestyle='--', label='Critical Level')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_title('Transformer Core Temperature Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = self.thermal_figure.add_subplot(2, 1, 2)
            ax2.plot(t_hours, load_profile * 100, 'g-', linewidth=2)
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Load (%)')
            ax2.set_title('Load Profile')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 120)

            self.thermal_figure.tight_layout()
            self.thermal_canvas.draw()

            # Show summary
            max_temp = np.max(T[:, 0])
            avg_temp = np.mean(T[:, 0])

            messagebox.showinfo("Thermal Analysis Complete",
                              f"Maximum Temperature: {max_temp:.2f}°C\n"
                              f"Average Temperature: {avg_temp:.2f}°C\n"
                              f"Temperature Rise: {max_temp - ambient_temp:.2f}°C")

        except Exception as e:
            messagebox.showerror("Thermal Simulation Error", str(e))

    def on_window_resize(self, event):
        """Handle window resize for auto-scaling"""
        # Only process resize events for the main window
        if event.widget == self.root:
            # Redraw all canvases to fit new size
            try:
                self.efficiency_canvas.draw()
                self.sim_canvas.draw()
                self.load_canvas.draw()
                self.thermal_canvas.draw()
                self.alt_canvas.draw()
                self.comp_canvas.draw()
            except:
                pass

    def save_results(self):
        """Save analysis results to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"transformer_analysis_{timestamp}.txt"

            with open(filename, 'w') as f:
                f.write("TRANSFORMER ANALYSIS RESULTS\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(self.results_text.get(1.0, tk.END))

            messagebox.showinfo("Success", f"Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """
Advanced Transformer Analysis Tool
Version 2.0

Features:
• All-day efficiency calculation
• Dynamic ODE simulation (RK45/Euler)
• Real-time load analysis
• Thermal behavior modeling
• Interactive visualization

Developed for Electrical Engineering Applications
        """
        messagebox.showinfo("About", about_text)

    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
USER GUIDE

1. All-Day Efficiency Tab:
   - Enter transformer parameters
   - Set operating schedule (must total 24 hours)
   - Click 'Calculate Efficiency' to see results

2. Dynamic Simulation Tab:
   - Select ODE solver (RK45 recommended)
   - Set simulation duration and time step
   - Click 'Start' to run simulation
   - View real-time results

3. Load Analysis Tab:
   - Adjust load parameters using sliders
   - Analysis updates automatically
   - View power triangle, efficiency curve, etc.

4. Thermal Analysis Tab:
   - Set thermal parameters
   - Click 'Simulate Thermal Behavior'
   - View temperature rise over time

Tips:
• Use RK45 solver for accurate results
• Smaller time steps = more accurate but slower
• All plots are auto-scaling and responsive
        """
        messagebox.showinfo("User Guide", guide_text)


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = TransformerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
