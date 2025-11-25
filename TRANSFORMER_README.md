# Advanced Transformer Analysis Tool

## Overview
A comprehensive Python + Tkinter application for electrical engineering transformer analysis, featuring real-time simulations, ODE solvers, and interactive visualizations.

## Problem Solution: All-Day Efficiency

### Given Problem:
A 100 kVA distribution transformer has:
- Full-load loss: 4 kW (equally divided between iron and copper)
- Iron loss: 2 kW
- Copper loss at full-load: 2 kW

**Operating Schedule (24 hours):**
- Full-load: 4 hours
- Half-load: 6 hours
- No-load (negligible output): 14 hours

### Solution:

#### Step 1: Calculate Energy Output
```
Full-load output = 100 kVA × 4 h = 400 kWh
Half-load output = 50 kVA × 6 h = 300 kWh
Total output = 400 + 300 = 700 kWh
```

#### Step 2: Calculate Energy Losses

**Iron Losses (constant):**
```
Iron losses = 2 kW × 24 h = 48 kWh
```

**Copper Losses (vary with load²):**
```
At full-load: 2 kW × 4 h = 8 kWh
At half-load: 2 × (0.5)² × 6 h = 2 × 0.25 × 6 = 3 kWh
At no-load: 0 kWh
Total copper losses = 8 + 3 = 11 kWh
```

**Total Losses:**
```
Total losses = 48 + 11 = 59 kWh
```

#### Step 3: Calculate All-Day Efficiency
```
Total input = Output + Losses = 700 + 59 = 759 kWh
All-day efficiency = (Output / Input) × 100
                   = (700 / 759) × 100
                   = 92.23%
```

### Answer: **92.23% All-Day Efficiency**

---

## Application Features

### 1. User Interface (Tkinter GUI)
- **Main Menu**: File, Simulation, and Help menus
- **Tabbed Interface**: 4 specialized analysis tabs
- **Input Parameters**: Easy-to-use entry fields and sliders
- **Control System**: Start, Stop, Reset buttons
- **Responsive Design**: Auto-scaling when window resizes

### 2. Calculation Modules

#### All-Day Efficiency Calculator
- Transformer parameter inputs
- Operating schedule configuration
- Detailed energy analysis
- Loss distribution breakdown

#### Dynamic Simulation
- Real-time ODE solving
- Magnetic flux simulation
- Current analysis (primary & secondary)
- Temperature dynamics

#### Load Analysis
- Interactive sliders for real-time control
- Load factor adjustment (0-120%)
- Power factor control
- Frequency variation (45-65 Hz)

#### Thermal Analysis
- Temperature rise calculation
- Cooling factor modeling
- Load cycle simulation
- Warning/critical level indicators

### 3. ODE Solvers

#### Euler Method
```python
y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
```
- Simple, fast
- Good for quick analysis
- Less accurate for stiff equations

#### Runge-Kutta 4th Order (RK45)
```python
k1 = dt * f(t, y)
k2 = dt * f(t + dt/2, y + k1/2)
k3 = dt * f(t + dt/2, y + k2/2)
k4 = dt * f(t + dt, y + k3)
y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
```
- High accuracy
- Recommended for precise simulations
- Better stability

### 4. Differential Equations

The transformer model includes:

1. **Flux Equation** (Faraday's Law):
   ```
   dΦ/dt = (V_applied - R_p × I_p) / N
   ```

2. **Primary Current**:
   ```
   dI_p/dt = (V - R_p×I_p - X_p×I_p) / τ
   ```

3. **Secondary Current**:
   ```
   dI_s/dt = -(R_s + R_load)×I_s / τ
   ```

4. **Temperature Dynamics**:
   ```
   dT/dt = (Q_generated - Q_dissipated) / C_thermal
   ```

### 5. Visualization Features

#### Real-time Plots:
- Magnetic flux vs time
- Primary & secondary currents
- Core temperature
- Power losses
- Efficiency curves
- Load profiles
- Power triangles
- Voltage regulation

#### Interactive Elements:
- Auto-scaling graphs
- Responsive to window resize
- Color-coded displays
- Value annotations
- Grid overlays

## Installation & Usage

### Requirements:
```bash
pip install numpy matplotlib
```

### Run the Application:
```bash
python3 transformer_analysis_advanced.py
```

### Quick Start Guide:

1. **All-Day Efficiency Tab**:
   - Enter transformer specs (default: 100 kVA)
   - Set operating hours (must total 24)
   - Click "Calculate Efficiency"
   - View detailed results and graphs

2. **Dynamic Simulation Tab**:
   - Choose solver: RK45 (recommended) or Euler
   - Set duration (default: 10s) and time step
   - Click "▶ Start" to run simulation
   - Watch real-time progress bar
   - View dynamic behavior graphs

3. **Load Analysis Tab**:
   - Adjust sliders for load factor, power factor, frequency
   - Analysis updates automatically
   - View power triangle, efficiency curve, voltage regulation
   - Check operating parameters table

4. **Thermal Analysis Tab**:
   - Set ambient temperature and cooling factor
   - Click "Simulate Thermal Behavior"
   - View temperature rise over 8-hour cycle
   - Check max/average temperatures

## Advanced Features for Electrical Engineering

### 1. Power System Analysis
- Power triangle visualization
- Real and reactive power calculation
- Power factor correction analysis
- Apparent power computation

### 2. Loss Analysis
- Iron loss (core/hysteresis)
- Copper loss (I²R)
- Load-dependent loss calculation
- Loss distribution pie charts

### 3. Efficiency Analysis
- All-day efficiency calculation
- Efficiency vs load curves
- Maximum efficiency point
- Loss percentage breakdown

### 4. Voltage Regulation
- Voltage drop calculation
- Load-dependent regulation
- Percentage regulation plots
- Secondary voltage variation

### 5. Thermal Management
- Temperature rise prediction
- Cooling analysis
- Thermal time constants
- Warning/critical level monitoring

### 6. Dynamic Behavior
- Transient response
- Steady-state analysis
- Magnetizing current
- Inrush current simulation

## Technical Specifications

### Transformer Model Parameters:
- **Rated Power**: 100 kVA (adjustable)
- **Voltage Ratio**: 11 kV / 415 V
- **Iron Loss**: 2 kW
- **Copper Loss**: 2 kW at full load
- **Resistances**: R₁ = 0.5Ω, R₂ = 0.01Ω
- **Reactances**: X₁ = 1.0Ω, X₂ = 0.02Ω

### Simulation Parameters:
- **Time Step**: 0.01s (adjustable)
- **Duration**: 10s (adjustable)
- **Frequency**: 50 Hz (adjustable 45-65 Hz)
- **Temperature Range**: 25-150°C

## Key Advantages

1. **Comprehensive Analysis**: All aspects in one tool
2. **Real-time Visualization**: See changes instantly
3. **Professional ODE Solvers**: Industry-standard methods
4. **Educational Value**: Perfect for learning
5. **Practical Applications**: Real-world transformer analysis
6. **No Syntax Errors**: Fully tested and validated
7. **Auto-scaling UI**: Responsive to window changes
8. **Multi-threaded**: Non-blocking simulations

## Practical Applications

1. **Transformer Design**: Optimize parameters
2. **Load Planning**: Analyze daily load cycles
3. **Efficiency Studies**: Find optimal operating points
4. **Thermal Management**: Prevent overheating
5. **Loss Reduction**: Identify loss sources
6. **Educational Tool**: Teaching transformer theory
7. **Research**: Dynamic behavior studies
8. **Maintenance Planning**: Predict failures

## Example Results

For the given problem:
```
Transformer: 100 kVA
Schedule: 4h full-load, 6h half-load, 14h no-load

Results:
- Total Output: 700 kWh
- Iron Losses: 48 kWh
- Copper Losses: 11 kWh
- Total Input: 759 kWh
- All-Day Efficiency: 92.23%
- Loss Percentage: 7.77%
```

## Future Enhancements

- [ ] Three-phase transformer analysis
- [ ] Harmonic analysis (FFT)
- [ ] Short circuit simulation
- [ ] Protection coordination
- [ ] Cost analysis
- [ ] Export to Excel/PDF
- [ ] Database integration
- [ ] Cloud synchronization

## License
Educational and research use

## Author
Advanced Electrical Engineering Analysis Tool
Created for comprehensive transformer analysis and education

---

**Note**: This tool combines theoretical accuracy with practical utility, making it ideal for both educational purposes and real-world transformer analysis in electrical engineering applications.
