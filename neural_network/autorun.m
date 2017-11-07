% ========== FUNCTION ============
% Used to train the neural network and save it to disk
clear;

% ========== VARIABLE DECLARATION ==========
training_ex_index_lower = 1;
training_ex_index_higher = 100; %max 7000
nodes_per_layer = [112 112 1];
learning_rate = 0.5;
regularization_term = 0;
max_iterations = 10000;
max_acceptable_error = 1.0000e-07; % max relative acceptable error between thetas

% ========== READ CSV ===========
csv_file = csvread('../datasets/one-hot-mushrooms.csv', 1, 0);
training_set = csv_file(training_ex_index_lower:training_ex_index_higher, :);
% Inpu
X = training_set(:, 3:end);
% Output
Y = training_set(:, 1);

[theta, err] = train(X, Y, nodes_per_layer, max_iterations, max_acceptable_error, learning_rate, regularization_term);

file_name = strcat('trained_networks/trained-', num2str(floor(time)));
file_name = strcat(file_name, '.mat');

save(file_name, 'theta');
