import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the neighbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):
        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def minkowski_distance(self, X, Y):
        distance = np.power(sum(np.power(np.absolute(X - Y), self.p)), 1 / self.p)
        return distance

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        m = len(self.data)
        n = x.shape[0]
        distance_matrix = np.zeros((n, m))

        for row in range(n):
            distance_matrix[row] = np.array([self.minkowski_distance(self.data[col], x[row]) for col in range(m)])

        return distance_matrix

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x

        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        neigh_dist = np.zeros((x.shape[0], self.k_neigh))
        idx_neigh = np.zeros((x.shape[0], self.k_neigh))
        distance_matrix = self.find_distance(x)
        for query_instance in range(len(x)):
            current_query = distance_matrix[query_instance]
            sorted_distance_indices = np.argsort(current_query)[:self.k_neigh]
            k_neighbours_distances = [current_query[i] for i in sorted_distance_indices]
            neigh_dist[query_instance] = np.array(k_neighbours_distances)
            idx_neigh[query_instance] = np.array(sorted_distance_indices)

        idx_neigh = idx_neigh.astype(np.int64)

        return neigh_dist, idx_neigh

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        distance, indices = self.k_neighbours(x)
        predicted = []
        for query_instance in range(len(indices)):
            if self.weighted:
                noise = 10 ** -8
                freq_list = {}
                for sample in range(self.k_neigh):
                    _class = self.target[indices[query_instance][sample]]
                    if _class not in freq_list:
                        freq_list[_class] = 0
                    freq_list[_class] = freq_list[_class] + 1 / (distance[query_instance][sample] + noise)
                prediction = max(freq_list, key=lambda _x: freq_list[_x])
                all_possible = [target_class for target_class, freq in freq_list.items() if
                                freq == freq_list[prediction]]
                prediction = min(all_possible)
            else:
                target_list = []
                for sample in range(len(indices[query_instance])):
                    target_list.append(self.target[indices[query_instance][sample]])
                val, counts = np.unique(target_list, return_counts=True)
                prediction = val[np.argmax(counts)]

            predicted.append(prediction)
        return np.array(predicted)

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        predicted = self.predict(x)
        true_result = 0
        for index in range(len(predicted)):
            if predicted[index] == y[index]:
                true_result += 1

        accuracy = true_result / len(x) * 100
        return accuracy
