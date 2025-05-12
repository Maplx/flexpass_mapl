import matplotlib.pyplot as plt
import numpy as np

# Data for each plot category
data = {
    "n_apps": {
        "x": [2, 4, 6, 8, 10, 12],
        "Time (s)": {
            "SMT": [41.1174, 163.8006, 459.3040, 1043.2792, 2567.6505, 3514.2390],
            "Heu": [0.2259, 0.455, 0.648, 0.8482, 1.0983, 1.2785]
        },
        "Flexibility": {
            "SMT": [2, 4, 6, 7.8287, 5.7601, 0.72],
            "Heu": [2.0, 3.8852, 5.0089, 5.1969, 4.3719, 2.4357]
        },
        "Success ratio (%)": {
            "SMT": [100, 100, 100, 98, 58, 6],
            "Heu": [100.0, 100.0, 100.0, 97.0, 79.0, 43.0]
        }
    },
    "n_states": {
        "x": [4, 6, 8, 10, 12, 14],
        "Time (s)": {
            "SMT": [246.8611, 284.2499, 385.9225, 641.9652, 1298.5417, 2659.6586],
            "Heu": [0.4889, 0.6178, 0.6679, 0.7877, 0.8556, 0.9493]
        },
        "Flexibility": {
            "SMT": [5, 5, 5, 5, 4.6, 2.1],
            "Heu": [4.7771, 4.6003, 4.282, 3.9147, 3.8073, 3.588]
        },
        "Success ratio (%)": {
            "SMT": [100, 100, 100, 100, 92, 42],
            "Heu": [100.0, 100.0, 100.0, 99.0, 100.0, 100.0]
        }
    },
    "n_flows": {
        "x": [4, 8, 12, 16, 20, 24],
        "Time (s)": {
            "SMT": [68.4774, 294.8798, 1015.0443, 2178.1569, 3064.5607, 3510.2169],
            "Heu": [0.6425, 0.6165, 0.5761, 0.6085, 0.6487, 0.6606]
        },
        "Flexibility": {
            "SMT": [5, 5, 5, 3.4804, 1.55, 0.35],
            "Heu": [4.9785, 4.6003, 3.6322, 2.9118, 1.8161, 0.9122]
        },
        "Success ratio (%)": {
            "SMT": [100, 100, 100, 70, 31, 7],
            "Heu": [100.0, 100.0, 98.0, 91.0, 74.0, 48.0]
        }
    },
    "n_links": {
        "x": [30, 26, 22, 18, 14, 10],
        "Time (s)": {
            "SMT": [290.6386, 279.1473, 357.9217, 422.7252, 938.0674, 2075.0972],
            "Heu": [0.5498, 0.4686, 0.4098, 0.3336, 0.2643, 0.1961]
        },
        "Flexibility": {
            "SMT": [5, 5, 4.9963, 4.9463, 4.5592, 2.7498],
            "Heu": [4.6003, 4.384, 4.0204, 3.6672, 2.6656, 1.7613]
        },
        "Success ratio (%)": {
            "SMT": [100, 100, 100, 99, 92, 56],
            "Heu": [100.0, 100.0, 100.0, 95.0, 82.0, 70.0]
        }
    },
    "n_slots": {
        "x": [40, 60, 80, 100, 120, 140],
        "Time (s)": {
            "SMT": [177.8538, 539.8638, 1067.9252, 2348.5193, 3139.4810, 3501.4151],
            "Heu": [0.4894, 0.6644, 0.8578, 1.0233, 1.2244, 1.4165]
        },
        "Flexibility": {
            "SMT": [5, 5, 5, 3.4804, 1.55, 0.35],
            "Heu": [4.6004, 4.736, 4.7206, 4.7247, 4.8338, 4.7474]
        },
        "Success ratio (%)": {
            "SMT": [100, 100, 100, 98, 58, 6],
            "Heu": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        }
    }
}
# Adjusting to access the labels correctly

# Plotting each category of data
metrics = ["Success ratio (%)", "Flexibility", "Time (s)"]
categories = ["n_apps", "n_states", "n_flows", "n_links", "n_slots"]
x_labels = {
    "n_apps": "Num of apps",
    "n_states": "Num of states",
    "n_flows": "Num of tasks",
    "n_links": "Num of links",
    "n_slots": "Num of slots"
}

fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)  # Increased figure size and resolution

# Map categories and metrics to the 2x2 grid
plot_combinations = [
    ("n_apps", "Flexibility", (0, 0), "(a)"),
    ("n_apps", "Time (s)", (1, 1), "(d)"),
    ("n_states", "Flexibility", (0, 1), "(b)"),
    ("n_flows", "Flexibility", (1, 0), "(c)"),
]

for category, metric, position, label in plot_combinations:
    row, col = position
    x = data[category]["x"]
    y_smt = data[category][metric]["SMT"]
    y_heu = data[category][metric]["Heu"]
    
    axs[row, col].plot(
        x, y_smt, marker='o', markersize=10, markerfacecolor='none', label="SMT", linewidth=2
    )
    axs[row, col].plot(
        x, y_heu, marker='^', markersize=10, markerfacecolor='none', label="SPaL", linewidth=2
    )
    
    axs[row, col].set_xlabel(x_labels[category], fontsize=26)
    axs[row, col].set_ylabel(metric, fontsize=26)
    axs[row, col].tick_params(axis='both', labelsize=22)  # Increase tick label size
    axs[row, col].set_xticks(x)  # Set x-axis to use integer ticks based on data
    axs[row, col].legend(fontsize=20)  # Increase legend text size
    axs[row, col].grid(True)
    axs[row, col].text(
        -0.1, 1.1, label, transform=axs[row, col].transAxes,
        fontsize=26, fontweight='bold', va='top', ha='left'
    )  # Add subplot labels

plt.subplots_adjust(wspace=0.2, hspace=0.2)  # Add more space between subplots
plt.tight_layout()
plt.show()
