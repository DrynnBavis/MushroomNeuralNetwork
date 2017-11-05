function [predicted, actual] = predict(x, y, theta)
  input_bias_unit = ones(size(x,1), 1);
  x = [input_bias_unit x];
  result = x * theta{1}';
  for i=2:size(theta, 2)
    bias_unit = 1;
    result = [bias_unit result];
    result = activation_func(result * theta{i}');
  end
  predicted = result;
  actual = y;
end
