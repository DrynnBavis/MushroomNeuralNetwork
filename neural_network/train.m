function [theta, err, cost_vector] = train(X, Y, nodes_per_layer, max_iterations, min_acceptable_error, learning_rate, regularization_term)
  rel_error = 1 + min_acceptable_error;
  num_layers = size(nodes_per_layer, 2);
  % Initialize thetas (weights between layers)
  theta = cell(1, size(nodes_per_layer, 2) - 1);
  cost_vector = [];

  % all layers except output layer have a bias unit associated with it
  for i=1:num_layers-1
      % Add an additional column for bias units
      % weights are initially random values from 0 - 1
      theta{i} = rand(nodes_per_layer(i+1), nodes_per_layer(i) + 1);
  end

  a = cell(1, size(nodes_per_layer, 2));
  z = cell(1, size(nodes_per_layer, 2));
  % delta{1} does not exist
  delta = cell(1, size(nodes_per_layer, 2));
  theta_changes = cell(1, size(nodes_per_layer, 2) - 1);
  iterations = 0;

  % ========== TRAINING ===========
  while(rel_error > min_acceptable_error && iterations < max_iterations)
      % ========== FORWARD PROPOGATION ==========
      % ========== LAYER ONE (input) ==========
      % Add bias unit to input
      bias_input = ones(size(X,1), 1);
      % a1 -> Output of layer 1
      a{1} = [bias_input X];

      % ========== OTHER LAYERS ==========
      for j=1:size(nodes_per_layer, 2) - 1
          z{j+1} = a{j} * theta{j}';
          a{j+1} = activation_func(z{j+1});
          if(j ~= size(nodes_per_layer, 2) - 1)
              bias_input = ones(size(a{j+1},1),1);
              a{j+1} = [bias_input a{j+1}];
          end
      end

      % ========== BACKWARD PROPOGATION ==========
      % ========== LAST LAYER (output) ==========
      % "Error" in last layer
      delE_deloutput = a{end} - Y;
      % This is g'(z(end))
      deloutput_delnet_output = activation_func_prime(a{end});
      % "Error" wrt net output
      delE_del_net_output = delE_deloutput .* deloutput_delnet_output;
      delta{end} = delE_del_net_output;

      % ========== OTHER LAYERS ==========
      for j=size(nodes_per_layer, 2) - 1:-1:2
          delz_after_dela_curr = theta{j}(:, 2:end);
          % This is g'(z(curr)) (10 x 3)
          dela_curr_delz_curr = activation_func_prime(z{j});
          % Multiply each error value by g'(z(curr)) and previous delta
          % delta(delE_del_z_after)
          delE_delz_curr = (delta{j+1} * delz_after_dela_curr) .* dela_curr_delz_curr;
          delta{j} = delE_delz_curr;
      end

      % ========== CALCULATE CHANGES TO WEIGHTS ==========
      % ========== WEGHTS (end-1) to end (theta{end-1}) ==========
      biggest_change = 0;
      for j=size(nodes_per_layer, 2) - 1:-1:1
         delta_theta{j} = delta{j+1}' * a{j}(:, 2:end);
         delta_theta_bias = sum(delta{j+1}, 1)';
         delta_theta{j} = [delta_theta_bias delta_theta{j}] ./ size(delta{j+1}, 1);
         % Add regularization to theta changess
         theta{j} = theta{j} - (delta_theta{j} + theta{j} .* regularization_term) .* learning_rate;

         if (max(max(delta_theta{j})) > biggest_change)
            biggest_change = max(max(abs(delta_theta{j})));
         end
      end
      rel_error = biggest_change;
      iterations = iterations + 1;
      cost_vector = [cost_vector, compute_cost(a{end}, Y)];
  end

  theta = theta;
  err = sum(abs((a{end} - Y) ./ Y)) / size(Y, 1);
end
