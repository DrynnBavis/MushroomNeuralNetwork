function g_x = activation_func(x)
    g_x = 1 ./ (1 + exp(-x));
end
