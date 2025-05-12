import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_matching(provision, loss, matching_request):
    # Prepare a list of valid requests and provisions
    valid_requests = []
    valid_provisions = []
    request_to_app = []
    provision_to_app = []

    # Collect all valid requests
    for req_app in range(len(matching_request)):
        for req in matching_request[req_app]:
            valid_requests.append(req)
            request_to_app.append(req_app)

    # Collect all valid provisions
    for prov_app in range(len(provision)):
        if provision[prov_app]:  # Only consider apps with provisions
            valid_provisions.append(provision[prov_app])
            provision_to_app.append(prov_app)

    # If there are no valid requests or provisions, return no matching
    if not valid_requests or not valid_provisions:
        return None, float('inf')  # Indicate no matching possible

    # Initialize the cost matrix for valid requests and provisions
    n_requests = len(valid_requests)
    n_provisions = len(valid_provisions)
    cost_matrix = np.full((n_requests, n_provisions), 9999.0)  # Large value for invalid matches

    # Populate the cost matrix
    for req_idx, (req_link, req_time_range) in enumerate(valid_requests):
        for prov_idx, prov_links in enumerate(valid_provisions):
            for prov_link, prov_time in prov_links:
                if prov_link == req_link and req_time_range[0] <= prov_time <= req_time_range[1]:
                    # Set the cost as the loss of the provider
                    cost_matrix[req_idx][prov_idx] = loss[provision_to_app[prov_idx]]

    # Apply the Hungarian algorithm to the valid part of the cost matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Collect the matching result
    matching = []
    total_cost = 0
    for row, col in zip(row_ind, col_ind):
        if cost_matrix[row, col] < 9999:  # Only valid matches
            matching.append((request_to_app[row], provision_to_app[col]))
            total_cost += cost_matrix[row, col]
        else:
            return None, float('inf')  # Indicate that a valid matching was not found

    return matching, total_cost


# Example usage:
provision = [[], [], [], [], [], [], [(14, 10)], [(14, 25), (14, 49), (14, 49)], [(14, 19)], [(14, 20), (14, 29), (14, 39), (14, 40)]]

loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3935047398480074, 0.0, 0.09413135111669925]

matching_request = [[(14, [10, 20]), (14, [20, 30]), (14, [25, 50]), (14, [30, 40]), (14, [40, 50])], [], [], [], [], [], [], [(14, [0, 25])], [], []]


matching, total_cost = hungarian_matching(provision, loss, matching_request)

if matching is None:
    print("No valid matching found")
else:
    print("Matching:", matching)
    print("Total Cost:", total_cost)


