function g_x_prime = activation_func_prime(x)
    g_x_prime = activation_func(x) .* (1 - activation_func(x));
end
