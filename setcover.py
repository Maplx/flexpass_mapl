import pulp


def weighted_set_cover_network(RR, RP, weights):
    # Create the LP problem
    prob = pulp.LpProblem("Weighted_Set_Cover", pulp.LpMinimize)

    # Create binary variables for each subset in RP
    x = {i: pulp.LpVariable(f"x{i}", cat='Binary') for i in range(len(RP))}

    # Objective function: minimize the total weight
    prob += 5*pulp.lpSum(x[i] for i in range(len(RP))) + pulp.lpSum(weights[i] * x[i] for i in range(len(RP)))

    # Constraints: each resource request must be covered at least once
    for e, T in RR:
        prob += pulp.lpSum(x[i] for i in range(len(RP))
                           if any((e, t) in RP[i] for t in range(T[0], T[1] + 1))) >= 1

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract the solution
    selected_subsets = [i for i in range(len(RP)) if x[i].value() == 1]
    total_weight = sum(weights[i] for i in selected_subsets)
    all_covered = all(
        any(any((e, t) in RP[i] for t in range(T[0], T[1] + 1)) for i in selected_subsets)
        for e, T in RR
    )

    return selected_subsets, total_weight, all_covered


# Example usage
RR = [ (3, [2, 5])]
RP = [
    {(1, 1), (2, 1), (3, 4)},
    {(1, 4),  (2, 1), (3, 1)},
    {(3, 3), (3, 4), (3, 5)}
]
loss = [8, 4, 3]

selected, total_loss, all_covered = weighted_set_cover_network(RR, RP, loss)

print("Set to cover:", RR)
if all_covered:
    print("Selected subsets:", selected)
    print("Total loss:", total_loss)
    print("Selected resource provisions:")
    for i in selected:
        print(f"RP_{i+1}: {RP[i]}")
else:
    print("unable to cover")
