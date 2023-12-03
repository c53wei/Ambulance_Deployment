import itertools

# filepath = "/Users/clarewei/Documents/BME/4A/BME411–Numerical Optimization /Final Project/Ambulance_Deployment.csv"
# weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# cartesian_product = itertools.product(weights, weights, weights, weights)
#
# with open(filepath, 'w') as csv:
#     for i in cartesian_product:
#         if sum(list(i)) == 1:
#             for j in i:
#                 csv.write(f'{j},')
#             csv.write('\n')


filepath = "/Users/clarewei/Documents/BME/4A/BME411–Numerical Optimization /Final Project/allocation_combinations.csv"
ambulances = list(range(49))
ervs = list(range(9))
cartesian_product = itertools.product(ambulances, ambulances, ervs, ervs)

with open(filepath, 'w') as csv:
    for i in cartesian_product:
        if (i[0] + i[1] <= 49) and (i[2] + i[3] <=8) and (0 not in i):
            for j in i:
                csv.write(f'{j},')
            csv.write('\n')