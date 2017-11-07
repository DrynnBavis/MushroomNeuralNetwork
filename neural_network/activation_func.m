function g_x = activation_func(x)
    g_x = 1.0 ./ (1.000009 + exp(-x));
end
