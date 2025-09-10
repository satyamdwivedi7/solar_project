# Solar PV Placement and Hydrogen Storage Sizing Optimization using ML Strategies

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PVLib](https://img.shields.io/badge/PVLib-0.10%2B-green.svg)](https://pvlib-python.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Abstract

This research addresses the critical challenge of optimizing solar photovoltaic (PV) systems integrated with hybrid hydrogen-battery energy storage to achieve maximum energy efficiency and cost-effectiveness while maintaining stringent safety standards. The study focuses on developing a comprehensive multi-objective optimization framework that considers three primary objectives:

1. **Maximizing energy yield** through optimal PV panel tilt angle configuration
2. **Minimizing total system costs** including equipment procurement, maintenance, and safety-related expenses  
3. **Ensuring positive energy balance** where energy stored equals energy produced minus energy consumed at all operational periods

The research methodology employs advanced optimization algorithms including machine learning approaches and metaheuristic techniques to solve the complex, multi-dimensional optimization problem. The system integrates photovoltaic panels with a hybrid energy storage system consisting of hydrogen fuel cells for long-term storage and conventional batteries for short-term energy management.

## 🎯 Key Features

- **Physics-Based Solar Simulation**: Uses PVLib for accurate solar irradiance and PV system modeling
- **Hybrid Storage Optimization**: Combined battery and hydrogen storage system simulation
- **Multi-Objective Optimization**: Genetic algorithms and machine learning for system optimization
- **Safety Constraint Modeling**: Hydrogen safety considerations and maintenance requirements
- **Real Weather Data**: NSRDB (National Solar Radiation Database) integration
- **Panel Placement Optimization**: Grid-based optimal positioning algorithms

## 🏗️ Project Structure

```
solar_pv_placement/
├── data/
│   ├── raw/
│   │   └── nsrdb.csv                    # NSRDB weather data
│   └── processed/
│       ├── pvlib_results.csv            # PVLib simulation results
│       └── solar_processed.csv          # Merged PV + weather data
├── src/
│   ├── pvlib_simulation.py              # Core PVLib solar simulation
│   ├── data_processing.py               # Data loading and processing
│   ├── energy_simulation.py             # Energy estimation pipeline
│   ├── panel_placement.py               # Panel placement optimization
│   ├── hydrogen_storage.py              # Hydrogen storage simulation
│   ├── hybrid_storage_simulation.py     # Combined battery + H2 storage
│   ├── optimization.py                  # System sizing optimization
│   └── optimizer_ga.py                  # Genetic algorithm optimization
├── reports/
│   ├── energy_estimation.csv            # Energy analysis results
│   ├── energy_profile.png               # Daily energy generation plots
│   ├── panel_placement.png              # Optimal placement visualization
│   ├── hydrogen_results.csv             # Hydrogen storage analysis
│   ├── hybrid_results.csv               # Hybrid storage simulation
│   └── optimization_results.csv         # System optimization results
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/satyamdwivedi7/solar_project.git
cd solar_pv_placement
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Simulation

1. **Run PVLib solar simulation**
```bash
cd src
python pvlib_simulation.py
```
This performs a tilt angle optimization sweep from 0° to 60° and identifies the optimal tilt angle.

2. **Process and merge data**
```bash
python data_processing.py
```

3. **Run energy simulations**
```bash
python energy_simulation.py
python panel_placement.py
```

4. **Run storage simulations**
```bash
python hydrogen_storage.py
python hybrid_storage_simulation.py
```

5. **Run optimization**
```bash
python optimization.py
python optimizer_ga.py
```

## 📊 Key Results

### Solar Energy Analysis
- **Location**: Chennai, India (13.05°N, 80.25°E)
- **Dataset**: 8,760 hourly records (full year 2014)
- **Non-zero energy hours**: 3,949 (45% of year)
- **Peak energy**: 1.928 kWh/hour (10-panel system)
- **Annual energy**: ~4,761 kWh (20-panel system)

### Optimal Configurations
- **Best tilt angle**: Determined through PVLib simulation sweep
- **Panel efficiency**: 18%
- **System sizes**: 5-50 kW analyzed
- **Storage capacity**: 50 kWh battery + 500 kWh H2 equivalent

## 🔧 Technical Implementation

### Solar Simulation Engine
The project uses **PVLib** for physics-accurate solar PV modeling including:
- Solar geometry calculations
- Temperature effects on panel efficiency  
- Spectral and angle-of-incidence losses
- Inverter efficiency modeling
- Weather-dependent performance

### Storage System Modeling
- **Battery Storage**: Lithium-ion with 90% roundtrip efficiency
- **Hydrogen Storage**: Electrolyzer (65% efficiency) + Fuel Cell (52% efficiency)
- **Safety Constraints**: H2 tank capacity limits and safety buffers
- **Hybrid Control**: Intelligent charge/discharge management

### Optimization Algorithms
- **Genetic Algorithm**: Multi-objective optimization using DEAP
- **Grid Search**: Systematic parameter space exploration
- **Machine Learning**: Feature-based optimization strategies

## 📈 Output Reports

The system generates comprehensive analysis reports:

### Energy Reports
- `energy_estimation.csv`: Annual/daily energy statistics
- `energy_profile.png`: Temporal energy generation patterns

### Optimization Reports  
- `optimization_results.csv`: System sizing recommendations
- `tilt_vs_energy.png`: Tilt angle optimization results

### Storage Analysis
- `hydrogen_results.csv`: H2 storage performance metrics
- `hybrid_results.csv`: Combined battery + H2 system analysis
- `hybrid_storage_summary.png`: Storage utilization visualization

### Panel Placement
- `panel_placement.png`: Optimal grid positioning visualization

## 🔬 Research Contributions

1. **Unified Optimization Framework**: Integrates PV placement, storage sizing, and safety constraints
2. **Physics-Based Modeling**: Uses industry-standard PVLib for accurate solar simulations
3. **Multi-Objective Approach**: Balances energy yield, cost, and safety considerations
4. **Hybrid Storage Innovation**: Optimizes combined battery + hydrogen storage systems
5. **Safety Integration**: Incorporates hydrogen safety protocols in optimization constraints

## 🎯 Expected Outcomes

- **15-30% efficiency improvement** over conventional fixed-angle PV installations
- **Significant cost reductions** through optimal component sizing
- **Enhanced system reliability** through integrated safety protocols
- **Holistic optimization** balancing performance, economics, and safety

## 🤝 Contributing

We welcome contributions to improve the optimization algorithms, add new storage technologies, or enhance the safety modeling. Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

### Key Dependencies
- **PVLib**: Solar PV system modeling library
- **NSRDB**: National Solar Radiation Database
- **DEAP**: Distributed Evolutionary Algorithms in Python
- **Pandas/NumPy**: Data processing and numerical computation
- **Matplotlib**: Visualization and plotting

### Research Applications
This framework supports research in:
- Renewable energy optimization
- Hybrid storage system design
- Multi-objective optimization algorithms
- Solar energy forecasting and planning
- Grid integration studies

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Satyam Dwivedi** - [satyamdwivedi7](https://github.com/satyamdwivedi7)

## 🆘 Support

For questions, issues, or collaboration opportunities:
- Create an issue in this repository
- Contact the author through GitHub

---

**Note**: This research contributes to renewable energy adoption by providing a holistic optimization approach that balances energy performance, economic viability, and safety considerations in hybrid PV-hydrogen systems, supporting both grid-connected and off-grid applications.
