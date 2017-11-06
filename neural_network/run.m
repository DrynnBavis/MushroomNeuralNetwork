clear;
% ========== VARIABLE DECLARATION ==========
training_ex_index_lower = 1;
training_ex_index_higher = 100; %max 7000
nodes_per_layer = [112 112 1];
learning_rate = 0.15;
regularization_term = 0;
max_iterations = 50000;
max_acceptable_error = 1.0000e-07; % max relative acceptable error between thetas

% ========== READ CSV ===========
csv_file = csvread('../datasets/one-hot-mushrooms.csv', 1, 0);
training_set = csv_file(training_ex_index_lower:training_ex_index_higher, :);
% Inpu
X = training_set(:, 3:end);
% Output
Y = training_set(:, 1);

% ========== START RUNNING ==========
printf('Welcome to the mushroom neural network predicter!\n')
train_model = input('Would you like to load an existing model? (y/n): ', 's');
if(strcmp(train_model, 'y'))
  while(exist("theta", "var") == 0)
    theta_file = input('Please input the csv filename: ', 's');
    try
      load(theta_file, 'theta');
    catch
      printf("Error, could not find file called %s\n\n", theta_file);
    end
  end
  err = find_error(theta) * 100;
  printf('Loaded %s.\nModel contains %d%% accuracy.\n\n', theta_file, err);
else
  printf('Training from scratch...\n');
  [theta, err] = train(X, Y, nodes_per_layer, max_iterations, max_acceptable_error, learning_rate, regularization_term);
  printf('Done training!\n');
end
err = find_error(theta) * 100;
printf('Model contains %d%% accuracy.\n\n', err);

current_session = true;
while(current_session)
  choice = input('What would you like to do? (1) predict, (2) save weights, (3) quit: ', 's');
  if(strcmp(choice, '1'))
    % row = randi(size(csv_file, 1) - training_ex_index_higher) + training_ex_index_higher;
    row = randi(training_ex_index_higher);
    x = csv_file(row, 3:end);
    y = csv_file(row, 1);
    printf('From row %d\n', row)
    [predicted, actual] = predict(x, y, theta);
    printf("actual: %d\n", actual);
    if(predicted > 0.5)
      printf("predicted: %d => 1\n", predicted);
    else
      printf("predicted: %d => 0\n", predicted);
    end
  elseif(strcmp(choice, '2'))
    weights_name = input('Enter the name of the file: ', 's');
    save(weights_name, 'theta');
  elseif(strcmp(choice, '3') || strcmp(choice, 'q'))
    printf('Quitting session\n');
    current_session = false;
  else
    printf('Please try again\n');
  end
end
