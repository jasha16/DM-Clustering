import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def read_data(filename):
    return pd.read_csv(filename, header=None, delimiter=" ")


def euclidean_distance(point1, point2):
    s = 0
    for (x, y) in zip(point1, point2):
        dist = (x - y) ** 2
        s += dist
    return s


def manhattan_distance(point1, point2):
    s = 0
    for (x, y) in zip(point1, point2):
        dist = abs(x-y)
        s += dist
    return s


def precision(actual, cluster):
    count = len(np.intersect1d(actual, cluster))
    return count / len(cluster)


def recall(actual, cluster):
    count = len(np.intersect1d(actual, cluster))
    return count / len(actual)


def fscore(p, r):
    return 2 * ((p * r) / (p + r))


def normalize(dataset):
    normalized_dataset = np.empty(dataset.shape)
    for i, row in enumerate(dataset):
        norm = np.linalg.norm(row)
        normalized_dataset[i] = (row / norm)
    return normalized_dataset


def find_label(d, actual):
    for i, a in enumerate(actual):
        if d in a:
            return i


def compute_clusters(modelType, maxk, dataset, actual, choice=None):
    average_r = []
    average_p = []
    average_fscore = []
    for k in range(1, maxk+1):

        model = modelType(dataset, k, choice)

        p = []
        r = []
        for i, cluster in enumerate(model.clusters):
            for x in cluster:
                label = find_label(x, actual)
                prec = precision(actual[label], cluster)
                rec = recall(actual[label], cluster)
                p.append(prec)
                r.append(rec)

        f = [fscore(prec, rec) for (prec, rec) in zip(p, r)]
        average_p.append(np.mean(p, axis=0))
        average_r.append(np.mean(r, axis=0))
        average_fscore.append(np.mean(f, axis=0))
    return np.round(average_p, 3), np.round(average_r, 3), np.round(average_fscore, 3)


def show_bcubed(p, r, f, title="", figname="figure"):
    plt.figure()
    plt.title(title)
    plt.plot(np.arange(len(p)) + 1, p, label="Precision")
    plt.plot(np.arange(len(r)) + 1, r, label="Recall")
    plt.plot(np.arange(len(f)) + 1, f, label="F-Score")
    plt.xlabel("Value of k")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(figname + ".png")


def print_table(p, r, f):
    print("{:<10} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}".format('k', 1, 2, 3, 4, 5, 6, 7, 8, 9))

    v1, v2, v3, v4, v5, v6, v7, v8, v9 = p
    print("{:<10} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}".format('Precision', v1, v2, v3, v4, v5, v6, v7,
                                                                                v8, v9))
    v1, v2, v3, v4, v5, v6, v7, v8, v9 = r
    print("{:<10} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}".format('Recall', v1, v2, v3, v4, v5, v6, v7,
                                                                                v8, v9))
    v1, v2, v3, v4, v5, v6, v7, v8, v9 = f
    print("{:<10} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}".format('F-Score', v1, v2, v3, v4, v5, v6, v7,
                                                                                v8, v9))


class Model:
    centroids = []
    data = []
    clusters = []

    def __init__(self, data, k, metric, choice=None):
        self.data = data
        self.k = k
        self.metric = metric
        self.choice = choice
        self.pick_centroids()

        self.train()

    def pick_centroids(self):
        centres = []
        for i in range(self.k):
            if i == 0:
                if self.choice is None:
                    choice = random.randint(0, len(self.data)-1)
                else:
                    choice = self.choice
                centres.append(self.data[choice])
            else:
                distance = np.zeros(self.data.shape[0])
                for c in centres:
                    distance_to_c = [self.metric(x, c) for x in self.data]
                    distance = distance_to_c + distance

                furthest = np.argmax(distance)
                centres.append(self.data[furthest])
        self.centroids = centres

    def train(self):
        for _ in range(20):
            clusters = self.clusters
            self.cluster()
            self.recompute_centres()
            new_clusters = self.clusters

            if clusters == new_clusters:
                break

    def cluster(self):
        new_clusters = []

        for _ in range(self.k):
            new_clusters.append([])

        for i, x in enumerate(self.data):
            distance_to_centroids = [self.metric(x, y) for y in self.centroids]
            closest_centre = np.argmin(distance_to_centroids)
            new_clusters[closest_centre].append(i)

        self.clusters = new_clusters

    def recompute_centres(self):
        pass


class KMeans(Model):
    def __init__(self, data, k, choice=None):
        super(KMeans, self).__init__(data, k, euclidean_distance, choice)

    def recompute_centres(self):
        for i, cluster in enumerate(self.clusters):
            cluster_values = self.data[cluster]
            if len(cluster_values) > 0:
                average = np.mean(cluster_values, axis=0)
                self.centroids[i] = average


class KMedians(Model):
    def __init__(self, data, k, choice=None):
        super(KMedians, self).__init__(data, k, manhattan_distance, choice)

    def recompute_centres(self):
        for i, cluster in enumerate(self.clusters):
            cluster_values = self.data[cluster]
            if len(cluster_values) > 0:
                medians = np.median(cluster_values, axis=0)
                self.centroids[i] = medians


if __name__ == '__main__':
    maxK = 9
    choice = 1
    categories = ["animals", "countries", "fruits", "veggies"]
    data = read_data(categories[0]).to_numpy()
    actual_ids = [np.arange(0, len(data))]
    prev_len = len(data)

    normalized = normalize(data[:, 1:])

    for i, x in enumerate(categories):
        if i == 0:
            continue
        else:
            curr = read_data(x).to_numpy()
            actual_ids.append(np.arange(prev_len, prev_len + len(curr)))
            prev_len += len(curr)
            data = np.concatenate((data, curr))

    p, r, f = compute_clusters(KMeans, maxK, data[:, 1:], actual_ids, choice)
    print("KMeans: ")
    print_table(p, r, f)
    show_bcubed(p, r, f, title="KMeans B-cubed evaluation", figname="means")

    p, r, f = compute_clusters(KMeans, maxK, normalize(data[:, 1:]), actual_ids, choice)
    print("KMeans Normalized: ")
    print_table(p, r, f)
    show_bcubed(p, r, f, title="KMeans Normalized B-cubed evaluation", figname="means_normal")

    p, r, f = compute_clusters(KMedians, maxK, data[:, 1:], actual_ids, choice)
    print("KMedians: ")
    print_table(p, r, f)
    show_bcubed(p, r, f, title="KMedians B-cubed evaluation", figname="medians")

    p, r, f = compute_clusters(KMedians, maxK, normalize(data[:, 1:]), actual_ids, choice)
    print("KMedians Normalized: ")
    print_table(p, r, f)
    show_bcubed(p, r, f, title="KMedians Normalized B-cubed evaluation", figname="medians_normal")
