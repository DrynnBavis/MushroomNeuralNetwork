% ========== PURPOSE ============
% Used to train the neural network and save it to disk
clear;
% # of training examples, learning rate, max iterations
nn_params = [
  % Change # of training sets
  % 200 0.01 100000;
  % 400 0.01 100000;
  % 600 0.01 100000;
  % 800 0.01 100000;
  % 1000 0.01 100000;
  % Change of learning rate
  200 0.01 100000;
  200 0.05 100000;
  200 0.07 100000;
  200 0.1 100000;
  200 0.15 100000;
  % Change max iteration
  200 0.07 5000;
  200 0.07 10000;
  200 0.07 100000;
  200 0.07 200000;
  200 0.07 300000;
  % Change best
  200 0.07 100000;
  200 0.07 100000;
  200 0.07 100000;
  200 0.07 100000;
  200 0.07 100000;
];

<<<<<<< HEAD
% ========== VARIABLE DECLARATION ==========
% Change these variables to manipulate the training
training_ex_index_lower = 1;       % initial row of csv file (min 1)
training_ex_index_higher = 1000;   % final row of csv file (max 7000)
num_training_ex = training_ex_index_higher - training_ex_index_lower + 1;
nodes_per_layer = [112 112 1];     % nodes in each layer
learning_rate = 0.07;              % learning rate (applied to delta changes)
regularization_term = 0;           % regularization term (for overfitting)
max_iterations = 100000;           % max iterations
min_acceptable_error = 1.0000e-07; % min relative acceptable error between thetas
=======
for i=1:size(nn_params, 1)
  % ========== VARIABLE DECLARATION ==========
  % Change these variables to manipulate the training
  training_ex_index_lower = 1;       % initial row of csv file (min 1)
  training_ex_index_higher = nn_params(i, 1);   % final row of csv file (max 7000)
  num_training_ex = training_ex_index_higher - training_ex_index_lower + 1;
  nodes_per_layer = [112 112 1];     % nodes in each layer
  learning_rate = nn_params(i, 2);              % learning rate (applied to delta changes)
  regularization_term = 0;           % regularization term (for overfitting)
  max_iterations = nn_params(i, 3);           % max iterations
  min_acceptable_error = 1.0000e-07; % min relative acceptable error between thetas
>>>>>>> 1f97c9706fa2b6cca742d172d950ee09191949e7

  % ========== READ CSV ===========
  csv_file = csvread('../datasets/one-hot-mushrooms.csv', 1, 0);
  training_set = csv_file(training_ex_index_lower:training_ex_index_higher, :);
  % Input
  X = training_set(:, 3:end);
  % Output
  Y = training_set(:, 1);

  tStart = tic;
  [theta, err, cost_vector] = train(X, Y, nodes_per_layer, max_iterations, min_acceptable_error, learning_rate, regularization_term);
  elapsedTime = toc(tStart);

  file_name = strcat('trained_networks/trained-', num2str(nn_params(i, 1)));
  file_name = strcat(file_name, num2str(nn_params(i, 2)));
  file_name = strcat(file_name, num2str(nn_params(i, 3)));
  file_name = strcat(file_name, '-');
  file_name = strcat(file_name, num2str(floor(time)));
  file_name = strcat(file_name, '.mat');

  save(file_name, 'theta', 'err', 'cost_vector', 'num_training_ex', 'nodes_per_layer', 'learning_rate', 'regularization_term', 'max_iterations', 'elapsedTime');
end
