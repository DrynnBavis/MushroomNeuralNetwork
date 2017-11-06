function error = find_error(theta)  
  csv_file = csvread('../datasets/correction-set.csv', 1, 0);
  correction_set = csv_file(:,:);
  hit = 0;
  miss = 0;
  for i = 1:1:1000
    % Input
    x = correction_set(i, 3:end);
    % Output
    y = correction_set(i, 1);
    [predicted, actual] = predict(x, y, theta);
    if((predicted > 0.5 && actual == 1) || (predicted <= 0.5 && actual ~= 1))
      hit = hit + 1;
    else
      miss = miss + 1;
    end
  end
  error = hit / 1000;
end