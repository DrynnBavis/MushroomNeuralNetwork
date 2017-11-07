function total_cost = compute_cost(predicted, actual)
  sum_of_costs = sum(actual.*log(predicted) + (1 - actual).*log(1 - predicted));
  s = size(predicted,1);
  total_cost = ((-1)*sum_of_costs) / s;
end
