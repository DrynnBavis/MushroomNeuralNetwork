clear;
% ========== READ CSV ===========
csv_file = csvread('or.csv', 1, 0);
% ========== VARIABLE DECLARATION ==========
input = csv_file(:, 1:end-1);
output = csv_file(:, end);
learning_rate = 0.1;
iterations = 10000;
nodes_per_layer = [3 3 1];

% Initialize thetas (weights between layers)
theta = cell(1, size(nodes_per_layer, 2) - 1);
% all layers except output layer have a bias unit associated with it
for i=1:size(nodes_per_layer,2)-1
    % Add an additional column for bias units
    % weights are initially random values from 0 - 1
    theta{i} = rand(nodes_per_layer(i+1), nodes_per_layer(i) + 1);
end

a = cell(1, size(nodes_per_layer, 2));
z = cell(1, size(nodes_per_layer, 2));
% delta{1} does not exist
delta = cell(1, size(nodes_per_layer, 2));
theta_changes = cell(1, size(nodes_per_layer, 2) - 1);

% ========== TRAINING ===========
for i=1:iterations
    % ========== FORWARD PROPOGATION ==========
    % ========== LAYER ONE (input) ==========
    % Add bias unit to input
    bias_input = ones(size(input,1), 1);
    % a1 -> Output of layer 1
    a{1} = [bias_input input];
    
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
    delE_deloutput = a{end} - output;
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
    for j=size(nodes_per_layer, 2) - 1:-1:1
       theta_changes{j} = delta{j+1}' * a{j}(:, 2:end) / size(delta{j+1}, 1);
       theta_curr_bias_changes = sum(delta{j+1}, 1)' / size(delta{j+1}, 1);
       theta_changes{j} = [theta_curr_bias_changes theta_changes{j}] .* learning_rate;
       theta{j} = theta{j} - theta_changes{j};
    end
    theta{1}
end

result = [a{end} output]
error = sum(abs((a{end} - output) ./ output)) / size(output,1)
error = (a{end}(end) - output(end)) ./ output(end) 

function g_x = activation_func(x)
    g_x = tanh(x);
end

function g_x_prime = activation_func_prime(x)
    g_x_prime = 1 - (activation_func(x).^2);% ones(size(x));
end