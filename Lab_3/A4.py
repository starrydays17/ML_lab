import math

def minkowski_distance(vector1, vector2, p):
    distance = 0
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p
    distance = distance ** (1 / p)
    return distance

A = [2, 4, 6, 8]
B = [1, 3, 5, 7]
p = int(input("Enter value of p: "))
distance = minkowski_distance(A, B, p)
print("Minkowski Distance =", distance)
if p == 1:
    print("This is Manhattan Distance.")
elif p == 2:
    print("This is Euclidean Distance.")