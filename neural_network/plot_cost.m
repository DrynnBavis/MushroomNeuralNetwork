function [] = plot_cost(x, y, plot_title, x_axis, y_axis)
  figure
  X = x;
  Y = y;
  plot(X, Y)
  title(plot_title)
  xlabel(x_axis)
  ylabel(y_axis)
end
