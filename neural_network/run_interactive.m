% ========== PURPOSE ============
% Used to interactively train/load neural networks
% One can check the accuracy of the training set and use it to predict against
% values in the csv file
clear;

% ========== VARIABLE DECLARATION ==========
% Change these variables to manipulate the training
training_ex_index_lower = 1;       % initial row of csv file (min 1)
training_ex_index_higher = 5000;   % final row of csv file (max 7000)
num_training_ex = training_ex_index_higher - training_ex_index_lower + 1;
nodes_per_layer = [112 112 1];     % nodes in each layer
learning_rate = 0.01;              % learning rate (applied to delta changes)
regularization_term = 0;           % regularization term (for overfitting)
max_iterations = 1000;           % max iterations
min_acceptable_error = 1.0000e-07; % min relative acceptable error between thetas
presentation_mode = false;
chosen_neural_network_file = 'trained_networks/trained-2000.07100000-1510296109.mat';

% ========== READ CSV ===========
csv_file = csvread('../datasets/one-hot-mushrooms.csv', 1, 0);
training_set = csv_file(training_ex_index_lower:training_ex_index_higher, :);
% Input
X = training_set(:, 3:end);
% Output
Y = training_set(:, 1);

% ========== START RUNNING ==========
fprintf('Welcome to the poisonous mushroom neural network predicter!\n')

% ========== PRESENTATION PURPOSES ==========
if(presentation_mode)
  fprintf('Presentation mode active.\n')
  fprintf('Please turn presentation_mode to false to load/train custom neural networks\n');
  fprintf('Welcome Professor Kwan/TAs!\n')
  neural_network_file = chosen_neural_network_file;
  load(neural_network_file);
  err = find_error(theta) * 100;
  fprintf('Loaded %s.\nModel contains %d%% accuracy.\n\n', neural_network_file, err);

% ========== TRAINING/LOADING CUSTOM NEURAL NETWORKS ===========
else
  train_model = input('Would you like to load an existing model? (y/n): ', 's');

  if(strcmp(train_model, 'y'))
    while(exist('theta', 'var') == 0)
      neural_network_file = input('Please input the csv filename: ', 's');
      try
        load(neural_network_file);
      catch
        fprintf('Error, could not find file called %s\n\n', neural_network_file);
      end
    end
    err = find_error(theta) * 100;
    fprintf('Loaded %s.\nModel contains %d%% accuracy.\n\n', neural_network_file, err);

  else
    fprintf('Training from scratch...\n');
    tStart = tic;
    [theta, err, cost_vector] = train(X, Y, nodes_per_layer, max_iterations, min_acceptable_error, learning_rate, regularization_term);
    elapsedTime = toc(tStart);
    fprintf('Done training!\n');
  end
end

iterations_performed = [1:1:size(cost_vector, 2)];
plot_cost(iterations_performed, cost_vector, 'cost vs iterations', 'iterations', 'cost');
current_session = true;
prediction_correct = 0;
prediction_count = 0;

while(current_session)
  choice = input('What would you like to do? (1) predict, (2) save weights, (3) quit: ', 's');
  if(strcmp(choice, '1'))
    row = randi(size(csv_file, 1) - training_ex_index_higher) + training_ex_index_higher;
    % row = randi(training_ex_index_higher);
    x = csv_file(row, 3:end);
    y = csv_file(row, 1);
    [predicted, actual] = predict(x, y, theta);
    rounded_predicted = round(predicted);
    prediction_count = prediction_count + 1;
    if(actual == rounded_predicted)
      prediction_correct = prediction_correct + 1;
    end

    fprintf('From row %d\n', row)
    fprintf('actual: %d\n', actual);
    fprintf('predicted: %f => %d\n', predicted, rounded_predicted);
    fprintf('Percentage correct %f\n', (prediction_correct/prediction_count) * 100);

  elseif(strcmp(choice, '2'))
    file_name = input('Enter the name of the file: ', 's');
    save(file_name, 'theta', 'err', 'cost_vector', 'num_training_ex', 'nodes_per_layer', 'learning_rate', 'regularization_term', 'max_iterations', 'elapsedTime');

  elseif(strcmp(choice, '3') || strcmp(choice, 'q'))
    fprintf('Quitting session\n');
    current_session = false;

  else
    fprintf('Please try again\n');
  end
end
