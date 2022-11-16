def clean_data(config, series, coef_mat, edges):
    series = series[:(-config.timesteps*2), :]
    edges = edges.astype('int')

    return series, config.beta_value, edges, coef_mat