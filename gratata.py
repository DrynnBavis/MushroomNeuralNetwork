import numpy as np

class Gratata:
    def __init__(self, hidden_layers, hidden_nodes, iterations, learning_rate):
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.iterations = iterations
        self.learning_rate = learning_rate

    def train(self, training_data):
        # training data should look like this:
        # [{ input: [], output: [] },{ input: [], output: [] }]
        # scrub will turn those arrays into numpy matrices

        formatted_data = self.format_data(training_data)
        self.initialize_weights(formatted_data)

        for i in range(self.iterations):
            results = self.forward_propogate(formatted_data)
            errors = self.back_propogate(formatted_data, results)

            print("==== RESULTS, iteration: " + str(i))
            print(results)
            print("===== ERRORS, iteration: " + str(i))
            print(errors)

    def predict(self, input):
        results = self.forward_propogate({ "input": np.array(input) })
        return results[-1]

    def format_data(self, data):
        input_data = []
        output_data = []

        for i in range(len(data)):
            input_data.append(data[i]["input"])
            output_data.append(data[i]["output"])

        formatted_data = {
            "input": np.array(input_data),
            "output": np.array(output_data)
        }

        return formatted_data

    def initialize_weights(self, data):
        # weights is a list of numpy arrays (matrices)
        # that store the connections between
        # input -> hidden layers -> hidden layers -> output

        self.weights = []

        # connections from input layer to hidden layer
        input_to_hidden_layer = np.random.uniform(size=(len(data["input"][0]), self.hidden_nodes))
        self.weights.append(input_to_hidden_layer)

        # connections between hidden layers
        for i in range(1, self.hidden_layers):
            hidden_to_hidden = np.random.uniform(size=(self.hidden_nodes, self.hidden_nodes))
            self.weights.append(hidden_to_hidden)

        # connections from last hidden layer to output layer
        hidden_to_output = np.random.uniform(size=(self.hidden_nodes, len(data["output"][0])))
        self.weights.append(hidden_to_output)

    def forward_propogate(self, data):
        # list of results of multiplying by weights and activating at each layer
        results = []

        # activation and sum between input layer and hidden layer
        input_to_hidden_sum = np.dot(data["input"], self.weights[0])
        input_to_hidden_activation = self.tanh(input_to_hidden_sum)
        results.append({"sum": input_to_hidden_sum, "activation": input_to_hidden_activation})

        # activation and sum between hidden layers
        for i in range(1, self.hidden_layers):
            hidden_to_hidden_sum = np.dot(results[i - 1]["activation"], self.weights[i])
            hidden_to_hidden_activation = self.tanh(hidden_to_hidden_sum)
            results.append({"sum": hidden_to_hidden_sum, "activation": hidden_to_hidden_activation})

        # last hidden layer to output layer
        hidden_to_output_sum = np.dot(results[-1]["activation"], self.weights[-1])
        hidden_to_output_activation = self.tanh(hidden_to_output_sum)
        results.append({"sum": hidden_to_output_sum, "activation": hidden_to_output_activation})

        return results

    def back_propogate(self, data, results):
        # take it back now yall, one hop this time! two hops this time!
        hidden_layers = self.hidden_layers
        learning_rate = self.hidden_layers
        weights = self.weights

        error = np.subtract(data["output"], results[-1]["activation"])

        # output layer to last hidden layer
        sum_at_layer = self.tanh_prime(results[-1]["sum"])
        transposed_activation_results = results[hidden_layers - 1]["activation"].transpose()

        delta = np.multiply(sum_at_layer, error) # element-wise multiplication
        changes = np.dot(transposed_activation_results, delta) * learning_rate
        weights[-1] = np.add(weights[-1], changes)

        # hidden layer to hidden layer
        for i in range(1, hidden_layers):
            sum_at_layer = self.tanh_prime(results[len(results) - (i + 1)]["sum"])
            transposed_activation_results = results[len(results) - (i + 1)]["activation"].transpose()

            delta = np.multiply(np.dot(delta, weights[len(weights) - i].transpose()), sum_at_layer)
            changes =  np.dot(transposed_activation_results, delta) * learning_rate
            weights[len(weights) - (i + 1)] = np.add(weights[len(weights) - (i + 1)], changes)

        # first hidden layer to input layer
        sum_at_layer = self.tanh_prime(results[0]["sum"])
        delta = np.multiply(np.dot(delta, weights[1].transpose()), sum_at_layer)
        changes = np.dot(data["input"].transpose(), delta) * learning_rate
        weights[0] = np.add(weights[0], changes)

        return error

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1.0 - np.tanh(x)**2


def OR_test():
    a = Gratata(2, 3, 100, 0.3)
    a.train([
        { "input": [0, 0, 0], "output": [0] },
        { "input": [0, 0, 1], "output": [1] },
        { "input": [0, 1, 0], "output": [1] },
        { "input": [0, 1, 1], "output": [1] },
        { "input": [1, 0, 0], "output": [1] },
        { "input": [1, 0, 1], "output": [1] },
        { "input": [1, 1, 0], "output": [1] }
    ])

    result = a.predict([1, 1, 1])
    print("=====THE PREDICTION====")
    print(result["activation"])

OR_test()

