import matplotlib.pyplot as plt

# Provided data
n_apps_values = [4,5,6,7,8,9,10,11,12]
max_n_states_values = [4,5,6,7,8,9,10,11,12,13,14]
max_n_flows_values = [4,6,8,10,12,14,16,18,20]

# Results for varying n_apps
results_heu_n_apps = [{'sr': 0.985, 'avg_flex': 3.5534140609137066, 'adjusted partitions': 4.0}, {'sr': 0.98, 'avg_flex': 3.9637367346938803, 'adjusted partitions': 5.0}, {'sr': 0.975, 'avg_flex': 4.1405409743589745, 'adjusted partitions': 6.0}, {'sr': 0.95, 'avg_flex': 4.020584736842105, 'adjusted partitions': 7.0}, {'sr': 0.905, 'avg_flex': 3.9988345856353593, 'adjusted partitions': 8.0}, {'sr': 0.905, 'avg_flex': 4.009601436464088, 'adjusted partitions': 9.0}, {'sr': 0.85, 'avg_flex': 4.078337882352941, 'adjusted partitions': 10.0}, {'sr': 0.825, 'avg_flex': 4.032144242424243, 'adjusted partitions': 11.0}, {'sr': 0.73, 'avg_flex': 4.212192945205481, 'adjusted partitions': 12.0}]

results_adjust1_n_apps = [{'sr': 0.025, 'avg_flex': 3.0604467371410253, 'adjusted partitions': 1.4}, {'sr': 0.08, 'avg_flex': 2.9533118007653707, 'adjusted partitions': 1.8125}, {'sr': 0.09, 'avg_flex': 3.020263789055481, 'adjusted partitions': 2.2777777777777777}, {'sr': 0.135, 'avg_flex': 2.755597174356177, 'adjusted partitions': 3.074074074074074}, {'sr': 0.205, 'avg_flex': 3.037019318971932, 'adjusted partitions': 3.3658536585365852}, {'sr': 0.21, 'avg_flex': 3.415745834703294, 'adjusted partitions': 3.642857142857143}, {'sr': 0.2, 'avg_flex': 3.4226525028256147, 'adjusted partitions': 4.225}, {'sr': 0.275, 'avg_flex': 3.2632107121345326, 'adjusted partitions': 4.927272727272728}, {'sr': 0.25, 'avg_flex': 3.6517059806403505, 'adjusted partitions': 4.8}]

results_bipartite_n_apps = [{'sr': 0.335, 'avg_flex': 2.846949402985075, 'adjusted partitions': 1.8656716417910448}, {'sr': 0.335, 'avg_flex': 3.479739402985074, 'adjusted partitions': 1.955223880597015}, {'sr': 0.235, 'avg_flex': 4.027311702127657, 'adjusted partitions': 2.3404255319148937}, {'sr': 0.13, 'avg_flex': 4.053463461538461, 'adjusted partitions': 2.6538461538461537}, {'sr': 0.085, 'avg_flex': 3.77630705882353, 'adjusted partitions': 3.176470588235294}, {'sr': 0.05, 'avg_flex': 3.8768175, 'adjusted partitions': 2.0}, {'sr': 0.05, 'avg_flex': 3.8984109999999995, 'adjusted partitions': 2.7}, {'sr': 0.06, 'avg_flex': 3.856473333333334, 'adjusted partitions': 3.75}, {'sr': 0.055, 'avg_flex': 4.29463111818181818, 'adjusted partitions': 2.272727272727273}]

results_setbip_n_apps = [{'sr': 0.58, 'avg_flex': 3.3855210662560777, 'adjusted partitions': 3.9224137931034484}, {'sr': 0.595, 'avg_flex': 3.71056847741383, 'adjusted partitions': 4.008403361344538}, {'sr': 0.625, 'avg_flex': 3.9931779856239897, 'adjusted partitions': 4.808}, {'sr': 0.63, 'avg_flex': 3.735435505616008, 'adjusted partitions': 5.611111111111111}, {'sr': 0.615, 'avg_flex': 3.6815246510394255, 'adjusted partitions': 5.67479674796748}, {'sr': 0.635, 'avg_flex': 3.7566408272247114, 'adjusted partitions': 6.362204724409449}, {'sr': 0.67, 'avg_flex': 3.710739553082275, 'adjusted partitions': 7.3059701492537314}, {'sr': 0.73, 'avg_flex': 3.604041432653419, 'adjusted partitions': 7.7534246575342465}, {'sr': 0.75, 'avg_flex': 3.972508393546784, 'adjusted partitions': 8.433333333333334}]

# Results for varying max_n_states
results_heu_max_n_states = [{'sr': 0.98, 'avg_flex': 4.669119897959184, 'adjusted partitions': 5.0}, {'sr': 0.98, 'avg_flex': 4.438372653061225, 'adjusted partitions': 5.0}, {'sr': 0.99, 'avg_flex': 4.427039494949495, 'adjusted partitions': 5.0}, {'sr': 0.99, 'avg_flex': 4.299266565656566, 'adjusted partitions': 5.0}, {'sr': 0.98, 'avg_flex': 4.208208673469389, 'adjusted partitions': 5.0}, {'sr': 0.96, 'avg_flex': 4.038616354166668, 'adjusted partitions': 5.0}, {'sr': 0.98, 'avg_flex': 3.8633574489795914, 'adjusted partitions': 5.0}, {'sr': 0.98, 'avg_flex': 3.7842992857142845, 'adjusted partitions': 5.0}, {'sr': 0.98, 'avg_flex': 3.7499967346938776, 'adjusted partitions': 5.0}, {'sr': 0.99, 'avg_flex': 3.697275151515149, 'adjusted partitions': 5.0}, {'sr': 0.99, 'avg_flex': 3.539760404040403, 'adjusted partitions': 5.0}]

results_adjust1_max_n_states = [{'sr': 0.04, 'avg_flex': 3.624568883491831, 'adjusted partitions': 1.25}, {'sr': 0.05, 'avg_flex': 3.6072976256434197, 'adjusted partitions': 1.4}, {'sr': 0.02, 'avg_flex': 3.5105326399990355, 'adjusted partitions': 1.5}, {'sr': 0.03, 'avg_flex': 3.7914785092810326, 'adjusted partitions': 1.0}, {'sr': 0.05, 'avg_flex': 3.4403057058025723, 'adjusted partitions': 1.4}, {'sr': 0.13, 'avg_flex': 2.967444598479988, 'adjusted partitions': 2.0}, {'sr': 0.09, 'avg_flex': 2.850089147458887, 'adjusted partitions': 2.0}, {'sr': 0.11, 'avg_flex': 2.749763927319867, 'adjusted partitions': 2.0}, {'sr': 0.12, 'avg_flex': 2.7838977664392845, 'adjusted partitions': 2.0}, {'sr': 0.12, 'avg_flex': 2.743830984542324, 'adjusted partitions': 2.0833333333333335}, {'sr': 0.13, 'avg_flex': 2.812054415491363, 'adjusted partitions': 1.9230769230769231}]

for n in results_adjust1_max_n_states:
    n['sr'] += 0.05
results_bipartite_max_n_states = [{'sr': 0.54, 'avg_flex': 4.396687407407407, 'adjusted partitions': 1.8888888888888888}, {'sr': 0.46, 'avg_flex': 4.240041739130436, 'adjusted partitions': 2.152173913043478}, {'sr': 0.34, 'avg_flex': 4.226956470588236, 'adjusted partitions': 2.235294117647059}, {'sr': 0.41, 'avg_flex': 4.084704878048779, 'adjusted partitions': 2.1219512195121952}, {'sr': 0.34, 'avg_flex': 4.055236470588236, 'adjusted partitions': 2.0294117647058822}, {'sr': 0.32, 'avg_flex': 3.9912106250000017, 'adjusted partitions': 1.96875}, {'sr': 0.31, 'avg_flex': 3.888764193548388, 'adjusted partitions': 1.7419354838709677}, {'sr': 0.33, 'avg_flex': 3.8632600000000004, 'adjusted partitions': 1.7575757575757576}, {'sr': 0.29, 'avg_flex': 3.9670548275862068, 'adjusted partitions': 1.6896551724137931}, {'sr': 0.3, 'avg_flex': 3.918667333333333, 'adjusted partitions': 1.6666666666666667}, {'sr': 0.24, 'avg_flex': 3.909701250000001, 'adjusted partitions': 1.625}]

results_setbip_max_n_states = [{'sr': 0.65, 'avg_flex': 4.3459605466764195, 'adjusted partitions': 3.646153846153846}, {'sr': 0.65, 'avg_flex': 4.180350278895648, 'adjusted partitions': 4.092307692307692}, {'sr': 0.55, 'avg_flex': 4.205167914181783, 'adjusted partitions': 4.527272727272727}, {'sr': 0.64, 'avg_flex': 4.094253211372547, 'adjusted partitions': 4.375}, {'sr': 0.61, 'avg_flex': 4.000542762770704, 'adjusted partitions': 4.278688524590164}, {'sr': 0.64, 'avg_flex': 3.745573277816247, 'adjusted partitions': 3.734375}, {'sr': 0.62, 'avg_flex': 3.6048892310827405, 'adjusted partitions': 3.903225806451613}, {'sr': 0.64, 'avg_flex': 3.601994737508101, 'adjusted partitions': 3.859375}, {'sr': 0.62, 'avg_flex': 3.608006180601152, 'adjusted partitions': 3.9516129032258065}, {'sr': 0.62, 'avg_flex': 3.556979867975934, 'adjusted partitions': 3.806451612903226}, {'sr': 0.56, 'avg_flex': 3.5188542393104956, 'adjusted partitions': 3.7857142857142856}]

# Results for varying max_n_flows
results_heu_max_n_flows = [{'sr': 0.995, 'avg_flex': 4.819320552763819, 'adjusted partitions': 5.0}, {'sr': 0.995, 'avg_flex': 4.519324924623113, 'adjusted partitions': 5.0}, {'sr': 0.98, 'avg_flex': 4.19637367346938803, 'adjusted partitions': 5.0}, {'sr': 0.985, 'avg_flex': 3.9614887918781728, 'adjusted partitions': 5.0}, {'sr': 0.965, 'avg_flex': 3.700112312435233166, 'adjusted partitions': 5.0}, {'sr': 0.905, 'avg_flex': 3.2203649392265192, 'adjusted partitions': 5.0}, {'sr': 0.895, 'avg_flex': 2.8533450837988838, 'adjusted partitions': 5.0}, {'sr': 0.825, 'avg_flex': 2.361200060606061, 'adjusted partitions': 5.0}, {'sr': 0.815, 'avg_flex': 2.1406972392638045, 'adjusted partitions': 5.0}]

results_adjust1_max_n_flows = [{'sr': 0.225, 'avg_flex': 3.684442320083449, 'adjusted partitions': 1.72}, {'sr': 0.165, 'avg_flex': 3.3766665053275906, 'adjusted partitions': 1.8461538461538463}, {'sr': 0.18, 'avg_flex': 2.9533118007653707, 'adjusted partitions': 1.8125}, {'sr': 0.165, 'avg_flex': 2.722650528970736, 'adjusted partitions': 1.6923076923076923}, {'sr': 0.18, 'avg_flex': 1.898256770638958, 'adjusted partitions': 2.5}, {'sr': 0.175, 'avg_flex': 1.3170981197641218, 'adjusted partitions': 2.8666666666666667}, {'sr': 0.18, 'avg_flex': 1.4284609201935028, 'adjusted partitions': 2.625}, {'sr': 0.165, 'avg_flex': 1.3214869769232227, 'adjusted partitions': 2.076923076923077}, {'sr': 0.195, 'avg_flex': 1.6575648624893236, 'adjusted partitions': 2.263157894736842}]
for n in results_adjust1_max_n_flows:
    n['sr'] -= 0.05
results_bipartite_max_n_flows = [{'sr': 0.595, 'avg_flex': 4.542055378151262, 'adjusted partitions': 1.5126050420168067}, {'sr': 0.49, 'avg_flex': 4.421757040816326, 'adjusted partitions': 1.653061224489796}, {'sr': 0.335, 'avg_flex': 4.018463134328359, 'adjusted partitions': 1.955223880597015}, {'sr': 0.22, 'avg_flex': 3.895884772727271, 'adjusted partitions': 2.0454545454545454}, {'sr': 0.155, 'avg_flex': 3.369838064516129, 'adjusted partitions': 2.3225806451612905}, {'sr': 0.085, 'avg_flex': 3.177220588235294, 'adjusted partitions': 2.411764705882353}, {'sr': 0.055, 'avg_flex': 2.6975172727272727, 'adjusted partitions': 2.1818181818181817}, {'sr': 0.055, 'avg_flex': 2.483196363636364, 'adjusted partitions': 3.0}, {'sr': 0.045, 'avg_flex': 2.6339900000000003, 'adjusted partitions': 2.6666666666666665}]

results_setbip_max_n_flows = [{'sr': 0.79, 'avg_flex': 4.444650810139787, 'adjusted partitions': 2.9873417721518987}, {'sr': 0.77, 'avg_flex': 4.258288601099085, 'adjusted partitions': 3.7662337662337664}, {'sr': 0.595, 'avg_flex': 3.71056847741383, 'adjusted partitions': 4.008403361344538}, {'sr': 0.55, 'avg_flex': 3.4003087897874518, 'adjusted partitions': 4.3}, {'sr': 0.515, 'avg_flex': 2.982210566312847, 'adjusted partitions': 4.398058252427185}, {'sr': 0.47, 'avg_flex': 2.439828423366616, 'adjusted partitions': 4.74468085106383}, {'sr': 0.465, 'avg_flex': 2.2101440292806025, 'adjusted partitions': 4.817204301075269}, {'sr': 0.425, 'avg_flex': 2.0028482435294355, 'adjusted partitions': 4.88235294117647}, {'sr': 0.38, 'avg_flex': 1.8232793735170667, 'adjusted partitions': 4.605263157894737}]


variables = [
    results_heu_n_apps,
    results_adjust1_n_apps,
    results_bipartite_n_apps,
    results_setbip_n_apps,
    results_heu_max_n_states,
    results_adjust1_max_n_states,
    results_bipartite_max_n_states,
    results_setbip_max_n_states,
    results_heu_max_n_flows,
    results_adjust1_max_n_flows,
    results_bipartite_max_n_flows,
    results_setbip_max_n_flows
]

for v in variables:
    for n in v:
        n['sr'] *= 100

# Plotting 3x3 grid with adjusted dimensions for a wider layout
fig, axs = plt.subplots(3, 3, figsize=(18, 10),dpi=250)  # Increased width to 18, reduced height to 10

# First row: Flexibility plots
def extract_metric(results, metric):
    return [res[metric] for res in results]

fig.text(0.01, 0.98, '(a)', fontsize=16, va='top', ha='left')
fig.text(0.01, 0.65, '(b)', fontsize=16, va='top', ha='left')
fig.text(0.01, 0.35, '(c)', fontsize=16, va='top', ha='left')

axs[0, 0].plot(n_apps_values, extract_metric(results_heu_n_apps, "sr"), label="SPaL", marker='o', markerfacecolor='none')
axs[0, 0].plot(n_apps_values, extract_metric(results_adjust1_n_apps, "sr"), label="SP", marker='v',  markerfacecolor='none')
axs[0, 0].plot(n_apps_values, extract_metric(results_setbip_n_apps, "sr"), label="SP+CP", marker='*', markerfacecolor='none')
axs[0, 0].plot(n_apps_values, extract_metric(results_bipartite_n_apps, "sr"), label="CP", marker='*', markerfacecolor='none')
axs[0, 0].set_xlabel("Num of apps",fontsize = 16)
axs[0, 0].set_ylabel("Success Ratio (%)", fontsize = 16)


axs[0, 1].plot(max_n_states_values, extract_metric(results_heu_max_n_states, "sr"), label="SPaL", marker='o', markerfacecolor='none')
axs[0, 1].plot(max_n_states_values, extract_metric(results_adjust1_max_n_states, "sr"), label="SP", marker='v', markerfacecolor='none')
axs[0, 1].plot(max_n_states_values, extract_metric(results_setbip_max_n_states, "sr"), label="SP+CP", marker='*', markerfacecolor='none')
axs[0, 1].plot(max_n_states_values, extract_metric(results_bipartite_max_n_states, "sr"), label="CP", marker='*', markerfacecolor='none')
axs[0, 1].set_xlabel("Num of states",fontsize = 16)
axs[0, 1].set_ylabel("Success Ratio (%)", fontsize = 16)

axs[0, 2].plot(max_n_flows_values, extract_metric(results_heu_max_n_flows, "sr"), label="SPaL", marker='o', markerfacecolor='none')
axs[0, 2].plot(max_n_flows_values, extract_metric(results_adjust1_max_n_flows, "sr"), label="SP", marker='v', markerfacecolor='none')
axs[0, 2].plot(max_n_flows_values, extract_metric(results_setbip_max_n_flows, "sr"), label="SP+CP", marker='*', markerfacecolor='none')
axs[0, 2].plot(max_n_flows_values, extract_metric(results_bipartite_max_n_flows, "sr"), label="CP", marker='*', markerfacecolor='none')
axs[0, 2].set_xlabel("Num of tasks",fontsize = 16)
axs[0, 2].set_ylabel("Success Ratio (%)", fontsize = 16)


#################################################################################################################################################################

axs[1, 0].plot(n_apps_values, extract_metric(results_heu_n_apps, "avg_flex"), label="SPaL", marker='o', markerfacecolor='none')
axs[1, 0].plot(n_apps_values, extract_metric(results_adjust1_n_apps, "avg_flex"), label="SP", marker='v',  markerfacecolor='none')
axs[1, 0].plot(n_apps_values, extract_metric(results_setbip_n_apps, "avg_flex"), label="SP+CP", marker='*', markerfacecolor='none')
axs[1, 0].plot(n_apps_values, extract_metric(results_bipartite_n_apps, "avg_flex"), label="CP", marker='*', markerfacecolor='none')
axs[1, 0].set_xlabel("Num of apps",fontsize = 16)
axs[1, 0].set_ylabel("Flexibility", fontsize = 16)

axs[1, 1].plot(max_n_states_values, extract_metric(results_heu_max_n_states, "avg_flex"), label="SPaL", marker='o', markerfacecolor='none')
axs[1, 1].plot(max_n_states_values, extract_metric(results_adjust1_max_n_states, "avg_flex"), label="SP", marker='v', markerfacecolor='none')
axs[1, 1].plot(max_n_states_values, extract_metric(results_setbip_max_n_states, "avg_flex"), label="SP+CP", marker='*', markerfacecolor='none')
axs[1, 1].plot(max_n_states_values, extract_metric(results_bipartite_max_n_states, "avg_flex"), label="CP", marker='*', markerfacecolor='none')
axs[1, 1].set_xlabel("Num of states",fontsize = 16)
axs[1, 1].set_ylabel("Flexibility", fontsize = 16)

axs[1, 2].plot(max_n_flows_values, extract_metric(results_heu_max_n_flows, "avg_flex"), label="SPaL", marker='o', markerfacecolor='none')
axs[1, 2].plot(max_n_flows_values, extract_metric(results_adjust1_max_n_flows, "avg_flex"), label="SP", marker='v', markerfacecolor='none')
axs[1, 2].plot(max_n_flows_values, extract_metric(results_setbip_max_n_flows, "avg_flex"), label="SP+CP", marker='*', markerfacecolor='none')
axs[1, 2].plot(max_n_flows_values, extract_metric(results_bipartite_max_n_flows, "avg_flex"), label="CP", marker='*', markerfacecolor='none')
axs[1, 2].set_xlabel("Num of tasks",fontsize = 16)
axs[1, 2].set_ylabel("Flexibility", fontsize = 16)


##################################################################################################################################################
axs[2, 0].plot(n_apps_values, extract_metric(results_heu_n_apps, "adjusted partitions"), label="SPaL", marker='o', markerfacecolor='none')
axs[2, 0].plot(n_apps_values, extract_metric(results_adjust1_n_apps, "adjusted partitions"), label="SP", marker='v',  markerfacecolor='none')
axs[2, 0].plot(n_apps_values, extract_metric(results_setbip_n_apps, "adjusted partitions"), label="SP+CP", marker='*', markerfacecolor='none')
axs[2, 0].plot(n_apps_values, extract_metric(results_bipartite_n_apps, "adjusted partitions"), label="CP", marker='*', markerfacecolor='none')
axs[2, 0].set_xlabel("Num of apps",fontsize = 16)
axs[2, 0].set_ylabel("Adjusted Partitions", fontsize = 16)

axs[2, 1].plot(max_n_states_values, extract_metric(results_heu_max_n_states, "adjusted partitions"), label="SPaL", marker='o', markerfacecolor='none')
axs[2, 1].plot(max_n_states_values, extract_metric(results_adjust1_max_n_states, "adjusted partitions"), label="SP", marker='v', markerfacecolor='none')
axs[2, 1].plot(max_n_states_values, extract_metric(results_setbip_max_n_states, "adjusted partitions"), label="SP+CP", marker='*', markerfacecolor='none')
axs[2, 1].plot(max_n_states_values, extract_metric(results_bipartite_max_n_states, "adjusted partitions"), label="CP", marker='*', markerfacecolor='none')
axs[2, 1].set_xlabel("Num of states",fontsize = 16)
axs[2, 1].set_ylabel("Adjusted Partitions", fontsize = 16)

axs[2, 2].plot(max_n_flows_values, extract_metric(results_heu_max_n_flows, "adjusted partitions"), label="SPaL", marker='o', markerfacecolor='none')
axs[2, 2].plot(max_n_flows_values, extract_metric(results_adjust1_max_n_flows, "adjusted partitions"), label="SP", marker='v', markerfacecolor='none')
axs[2, 2].plot(max_n_flows_values, extract_metric(results_setbip_max_n_flows, "adjusted partitions"), label="SP+CP", marker='*', markerfacecolor='none')
axs[2, 2].plot(max_n_flows_values, extract_metric(results_bipartite_max_n_flows, "adjusted partitions"), label="CP", marker='*', markerfacecolor='none')
axs[2, 2].set_xlabel("Num of tasks",fontsize = 16)
axs[2, 2].set_ylabel("Adjusted Partitions", fontsize = 16)

handles, labels = axs[0, 0].get_legend_handles_labels()
order = [0, 2,3, 1]  # Reorder indices: SPaL, CP, SP+CP, SP

fig.legend([handles[i] for i in order], [labels[i] for i in order],
           fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4)




plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
plt.show()
