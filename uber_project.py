

import numpy as np
import matplotlib.pyplot as plt
import csv


# Simple Driver class to be used for each driver
class Driver(object):
    def __init__(self, pos):
        self.time_left = 0
        self.position = pos


# Reads a csv file and returns the data is a 2D array
def read_file(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = np.int_(list(csv_reader))
    return data


# Floyd-Warshall algorithm for computing the shortest paths between all nodes
def floyd_warshall(graph, inf=float("inf")):
    N = len(graph)
    dist = np.ones((N, N), dtype=np.int) * inf
    for i in range(N):
        for j in range(N):
            if graph[i][j] != 0 and i != j:
                dist[i][j] = graph[i][j]
            elif i == j:
                dist[i][j] = 0
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


# An algorithm that computes the most central nodes, in terms of Closeness Centrality
# Returns the sorted indices of the nodes, from most central to least central
def closeness(graph):
    N = len(graph)
    closeness_array = []
    for i in range(N):
        closeness_array.append((i, sum(graph[i])))
    v1, v2 = zip(*sorted(closeness_array, key=lambda x: x[1]))
    return v1


# An algorithm to send a driver to a more central node
def send_drivers(drivers, distances, closeness_array):
    set_of_currents = set()
    list2 = []
    for j in range(len(drivers)):
        if drivers[j].time_left <= 1:
            set_of_currents.add(drivers[j].position)
            if drivers[j].time_left == 0:
                list2.append(j)
    for k in list2:
        ind_current = closeness_array.index(drivers[k].position)
        i = 0
        while i < ind_current:
            if distances[drivers[k].position][closeness_array[i]] == 1 and closeness_array[i] not in set_of_currents:
                if drivers[k].position in set_of_currents: # In case two drivers ended at same location
                    set_of_currents.remove(drivers[k].position)
                drivers[k].position = closeness_array[i]
                drivers[k].time_left = 1
                set_of_currents.add(closeness_array[i])
                break
            i += 1


# Given a list of requests and a city map, sends optimal Uber drivers to pickup locations
# Includes boolean parameters for whether to use closeness centrality and re-distribute. By default set to True
def optimize_pickups(requests, dist, num_drivers, bool_closeness=True, bool_distribute=True, inf=float("inf")):
    num_requests = len(requests)
    d = []
    if bool_closeness:
        close_array = closeness(dist)
    else:
        close_array = range(0, len(dist))
    for i in range(num_drivers):
        d.append(Driver(pos=close_array[i % 50]))
    total_wait_time = 0
    for k in range(num_requests):
        min_dist = inf
        min_dist_ind = -1
        for w in range(num_drivers):
            if k != 0:
                d[w].time_left = d[w].time_left - (requests[k][0]-requests[k-1][0])
            if d[w].time_left < 0:
                d[w].time_left = 0
            driver_dist = dist[d[w].position][requests[k][1]-1] + d[w].time_left
            if driver_dist == min_dist: # Tie-breaker: use closeness centrality
                ind_new = close_array.index(d[w].position)
                ind_old = close_array.index(d[min_dist_ind].position)
                if ind_new > ind_old:
                    min_dist_ind = w
            if driver_dist < min_dist:
                min_dist = np.int_(driver_dist)
                min_dist_ind = w
        d[min_dist_ind].position = requests[k][2]-1
        d[min_dist_ind].time_left = min_dist + dist[requests[k][1]-1][requests[k][2]-1]
        total_wait_time += min_dist
        if bool_distribute:
            send_drivers(d, dist, close_array)
    return total_wait_time


# Plots the amount of Uber drivers vs the total wait times
def plot_times(wait, number, i):
    plt.figure(i)
    for j in range(len(wait)):
        plt.scatter(range(2, number + 1), wait[j][1:])
    plt.title('Number of Uber Drivers vs Total Wait Time')
    plt.xlabel('Number of Uber Drivers')
    plt.ylabel('Total Wait Time')


requests_data = read_file('requests.csv')
requests_data2 = read_file('supplementpickups.csv')
network_data = read_file('network.csv')
dist_mat = floyd_warshall(network_data)


loops = 30

# Makes Figure 1
wait_times = np.zeros((2, loops), dtype=np.int)
for uber_drivers in range(1, loops+1):
    wait_times[0][uber_drivers - 1] = (optimize_pickups(requests_data, dist_mat, uber_drivers, False, False))
    wait_times[1][uber_drivers - 1] = (optimize_pickups(requests_data, dist_mat, uber_drivers, True, False))
plot_times(wait_times, loops, 1)
plt.legend(["No optimization attempted", "With closeness centrality"])

# Makes Figure 2
for uber_drivers in range(1, loops+1):
    wait_times[0][uber_drivers - 1] = (optimize_pickups(requests_data2, dist_mat, uber_drivers, False, False))
    wait_times[1][uber_drivers - 1] = (optimize_pickups(requests_data2, dist_mat, uber_drivers, True, False))
plot_times(wait_times, loops, 2)
plt.legend(["No optimization attempted", "With closeness centrality"])

# Makes Figure 3
wait_times = np.zeros((3, loops), dtype=np.int)
for uber_drivers in range(1, loops+1):
    wait_times[0][uber_drivers - 1] = (optimize_pickups(requests_data, dist_mat, uber_drivers, False, False))
    wait_times[1][uber_drivers-1] = (optimize_pickups(requests_data, dist_mat, uber_drivers, True, False))
    wait_times[2][uber_drivers-1] = (optimize_pickups(requests_data, dist_mat, uber_drivers))
plot_times(wait_times, loops, 3)
plt.legend(["No optimization attempted", "With closeness centrality", "With re-distribution"])

# Makes Figure 4
wait_times = np.zeros((3, loops), dtype=np.int)
for uber_drivers in range(1, loops+1):
    wait_times[0][uber_drivers - 1] = (optimize_pickups(requests_data2, dist_mat, uber_drivers, False, False))
    wait_times[1][uber_drivers-1] = (optimize_pickups(requests_data2, dist_mat, uber_drivers, True, False))
    wait_times[2][uber_drivers-1] = (optimize_pickups(requests_data2, dist_mat, uber_drivers))
plot_times(wait_times, loops, 4)
plt.legend(["No optimization attempted", "With closeness centrality", "With re-distribution"])
plt.show()










