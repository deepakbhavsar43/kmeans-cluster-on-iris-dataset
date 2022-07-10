import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def min_max(self):
    xmin = min(df['sepal_length'])
    ymin = min(df['sepal_width'])
    xmax = max(df['sepal_length'])
    ymax = max(df['sepal_width'])
    return xmin, ymin, xmax, ymax


df = pd.read_csv('Iris_Dataset/iris.csv')
xmin, ymin, xmax, ymax = min_max(df)
print(xmin, ymin, xmax, ymax)
k = 3
centroids = {
    i + 1: [np.random.randint(xmin, xmax), np.random.randint(ymin, ymax)] for i in range(k)
}
print(centroids)
# To visualize data-points and initial centroids
plt.scatter(df["sepal_length"], df["sepal_width"])
colmap = {1: 'red', 2: 'green', 3: 'blue'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], edgecolors='black')
plt.xlim(4, 8)
plt.ylim(1.5, 5)
plt.show()

def assignment1(df, centroids):
    for i in centroids.keys():
        df[f'distance_from_{i}'] = (
            np.sqrt(
                (df['sepal_length'] - centroids[i][0]) ** 2 + (df['sepal_width'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df


df = assignment1(df, centroids)

plt.scatter(df['sepal_length'], df['sepal_width'], color=df["color"])
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='X')
plt.xlim(4, 8)
plt.ylim(1.5, 5)
plt.show()

old_cent = centroids.copy()
print(centroids)
def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['sepal_length'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['sepal_width'])
    return k


centroids = update(centroids)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['sepal_length'], df['sepal_width'], color=df['color'])
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='x')
plt.xlim(4, 8)
plt.ylim(1.5, 5)
# for i in old_cent.keys():
#     old_x = old_cent[i][0]
#     old_y = old_cent[i][1]
#     dx = (centroids[i][0] - old_cent[i][0] * 0.75)
#     dy = (centroids[i][1] - old_cent[i][0] * 0.75)
#     ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()

df = assignment1(df, centroids)
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['sepal_length'], df['sepal_width'], color=df['color'])
co = 10
while co>0:
    df = assignment1(df, centroids)
    centroids = update(centroids)


for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='x')
plt.xlim(4, 8)
plt.ylim(1.5, 5)
plt.show()



