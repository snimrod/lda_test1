# Convert array of probabilities to array of locations (location in the list when ordered)
# gets a list of values and return a list where the value represent how many items in the list this item is bigger than
# the smallest value will be changed to 0 and biggest value to (size - 1)
def convert_locations(p_list):
    size = len(p_list)
    x = []
    ti = 0
    for n in p_list:
        x.append([n, ti])
        ti = ti + 1
    x.sort()
    loc = [0] * size

    for tn in range(1, size):
        if x[tn][0] == x[tn - 1][0]:
            loc[x[tn][1]] = loc[x[tn - 1][1]]
        else:
            loc[x[tn][1]] = tn

    return loc
