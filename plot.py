import numpy as np
import matplotlib.pyplot as plt

# Example dictionary for bipartite matching dynamic adjustment result
bipartite_result = {'n_adjusted_apps': [0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, -6, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, -6, 2, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'flex': [3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.44115, 3.44115, 3.44115, 3.44115, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 1.7556978531029053, 1.7556978531029053, 1.7556978531029053, 1.7556978531029053, 1.7556978531029053, 1.7556978531029053, 1.7556978531029053, 1.7556978531029053, 1.7556978531029053, 3.67343, 3.67343, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.52014, 3.52014, 3.52014, 3.52014, 3.52014, 3.52014, 3.52014, 3.64993, 3.64993, 3.64993, 3.64993, 3.64993, 3.64993, 3.64993, 3.64993, 3.64993, 3.64993, 3.66199, 3.66199, 3.66199, 3.66199, 3.66199, 3.66199, 3.66199, 3.66199, -0.6372012877886055, -0.6372012877886055, -0.6372012877886055, -0.6372012877886055, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544, 3.84544], 'time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.44551682472229004, 0, 0, 0, 0.4246644973754883, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005953311920166016, 0, 0, 0.0004985332489013672, 0, 0, 0, 0, 0, 0.4253673553466797, 0, 0.4490091800689697, 0, 0, 0, 0, 0, 0, 0.467909574508667, 0, 0, 0, 0, 0, 0, 0.42468857765197754, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.449507474899292, 0, 0, 0, 0, 0, 0, 0.449507474899292, 0.0019888877868652344, 0, 0, 0, 0.4539647102355957, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'reconfig_count': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 'xAxis': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]}
# Example dictionary for set cover dynamic adjustment result
set_cover_result = {'n_adjusted_apps': [0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, -6, -6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, -6, 0, 0, 0, -6, 0, -6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0], 'flex': [3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.44115, 3.44115, 3.44115, 3.44115, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.76004, 3.76004, 3.76004, 3.5894, 3.62385, 3.62385, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.31612, 3.31612, 3.31612, 3.31612, 3.31612, 3.31612, 3.67261, 3.67261, 3.67261, 3.67261, 3.85015, 3.85015, 3.26452, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.59826, 3.59826, 3.59826], 'time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.42676424980163574, 0, 0, 0, 0.4238154888153076, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.43360018730163574, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.46392321586608887, 0, 0, 0.6067459583282471, 0.48722124099731445, 0, 0.4415717124938965, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.46140432357788086, 0, 0, 0, 0, 0, 0.44256067276000977, 0, 0, 0, 0.440138578414917, 0, 0.43512821197509766, 0.46146535873413086, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5630888938903809, 0, 0], 'reconfig_count': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13], 'xAxis': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]}
heu_result  = {'n_adjusted_apps': [0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, -6, -6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, -6, 0, 0, 0, -6, 0, -6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0], 'flex': [3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.76004, 3.44115, 3.44115, 3.44115, 3.44115, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.51169, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.26452, 3.76004, 3.76004, 3.76004, 3.5894, 3.62385, 3.62385, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.31612, 3.31612, 3.31612, 3.31612, 3.31612, 3.31612, 3.67261, 3.67261, 3.67261, 3.67261, 3.85015, 3.85015, 3.26452, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.66435, 3.59826, 3.59826, 3.59826], 'time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6176497936248779, 0, 0, 0, 0.5546689033508301, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.49861574172973633, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.768496036529541, 0, 0, 0.47283029556274414, 0.5139880180358887, 0, 0.4534883499145508, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4534482955932617, 0, 0, 0, 0, 0, 0.4747960567474365, 0, 0, 0, 0.444500207901001, 0, 0.5670673847198486, 0.4311401844024658, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4192521572113037, 0, 0], 'reconfig_count': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13], 'xAxis': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]}

# Determine the minimum length between the three results
min_length = min(len(bipartite_result['flex']), len(set_cover_result['flex']), len(heu_result['flex']))

# Slice the lists to the minimum length
bipartite_flex = bipartite_result['flex'][:min_length]
bipartite_time = bipartite_result['time'][:min_length]
bipartite_reconfig = bipartite_result['reconfig_count'][:min_length]
bipartite_adjusted_apps = bipartite_result['n_adjusted_apps'][:min_length]

set_cover_flex = set_cover_result['flex'][:min_length]
set_cover_time = set_cover_result['time'][:min_length]
set_cover_reconfig = set_cover_result['reconfig_count'][:min_length]
set_cover_adjusted_apps = set_cover_result['n_adjusted_apps'][:min_length]

heu_flex = heu_result['flex'][:min_length]
heu_time = heu_result['time'][:min_length]
heu_reconfig = heu_result['reconfig_count'][:min_length]
heu_adjusted_apps = heu_result['n_adjusted_apps'][:min_length]

# Remove zeros for calculating adjusted applications
bipartite_adjusted_apps_no_zeros = [x for x in bipartite_adjusted_apps if x != 0]
set_cover_adjusted_apps_no_zeros = [x for x in set_cover_adjusted_apps if x != 0]
heu_adjusted_apps_no_zeros = [x for x in heu_adjusted_apps if x != 0]

# Calculate averages for flexibility, time, and adjusted apps without zeros
avg_bipartite_flex = np.mean(bipartite_flex)
avg_bipartite_time = np.mean(bipartite_time)
avg_bipartite_adjusted_apps = np.mean(bipartite_adjusted_apps_no_zeros) if bipartite_adjusted_apps_no_zeros else 0

avg_set_cover_flex = np.mean(set_cover_flex)
avg_set_cover_time = np.mean(set_cover_time)
avg_set_cover_adjusted_apps = np.mean(set_cover_adjusted_apps_no_zeros) if set_cover_adjusted_apps_no_zeros else 0

avg_heu_flex = np.mean(heu_flex)
avg_heu_time = np.mean(heu_time)
avg_heu_adjusted_apps = np.mean(heu_adjusted_apps_no_zeros) if heu_adjusted_apps_no_zeros else 0

# Function to sum the maximum reconfiguration counts
def sum_max_reconfig(reconfig_list):
    max_value = 0
    current_max = 0
    
    for r in reconfig_list:
        if r > 0:
            current_max = max(current_max, r)  # Track the maximum value in a contiguous sequence
        else:
            max_value += current_max  # Add the current max and reset
            current_max = 0

    max_value += current_max  # Add any remaining max value in the list
    return max_value

# Reconfiguration Count: sum of max values for each algorithm
bipartite_total_reconfig = sum_max_reconfig(bipartite_reconfig)
set_cover_total_reconfig = sum_max_reconfig(set_cover_reconfig)
heu_total_reconfig = sum_max_reconfig(heu_reconfig)

# Reconfigurations per 100 iterations (to normalize across iterations)
avg_bipartite_reconfig_per_100 = (bipartite_total_reconfig / min_length) * 100
avg_set_cover_reconfig_per_100 = (set_cover_total_reconfig / min_length) * 100
avg_heu_reconfig_per_100 = (heu_total_reconfig / min_length) * 100

# Create figure with 4 subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

# Flexibility bar plot
rects1 = ax1.bar(['Bipartite', 'Set Cover', 'Reconfiguration'], [avg_bipartite_flex, avg_set_cover_flex, avg_heu_flex], color=['orange', 'blue', 'green'])
ax1.set_title('Average System Flexibility')
ax1.set_ylabel('Flexibility')

# Time bar plot
rects2 = ax2.bar(['Bipartite', 'Set Cover', 'Reconfiguration'], [avg_bipartite_time, avg_set_cover_time, avg_heu_time], color=['orange', 'blue', 'green'])
ax2.set_title('Average Execution Overhead')
ax2.set_ylabel('Time (s)')

# Reconfiguration Count per 100 iterations bar plot (Summing max values)
rects3 = ax3.bar(['Bipartite', 'Set Cover', 'Reconfiguration'], [avg_bipartite_reconfig_per_100,avg_set_cover_reconfig_per_100, avg_heu_reconfig_per_100], color=['orange', 'blue', 'green'])
ax3.set_title('Reconfigurations per 100 State Transitions')
ax3.set_ylabel('Reconfigurations')

# Adjusted applications bar plot
rects4 = ax4.bar(['Bipartite', 'Set Cover', 'Reconfiguration'], [avg_bipartite_adjusted_apps, avg_set_cover_adjusted_apps, avg_heu_adjusted_apps], color=['orange', 'blue', 'green'])
ax4.set_title('Average Number of Applications Involved')
ax4.set_ylabel('Number of Applications')

# Function to label the bars with their heights
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add labels to all bar plots
autolabel(rects1, ax1)
autolabel(rects2, ax2)
autolabel(rects3, ax3)
autolabel(rects4, ax4)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
