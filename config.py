ncmapss_path = 'data/N_CMAPSS/N-CMAPSS_DS08a-009.h5'
regime = 0
train_units = [1, 2, 3]
test_units = [10]
cycles_frac = 0.1
model_params = {
    'indim': 18,
    'hidden_dim': 100,
    'obsdim': 10,
    'outdim': 14,
    'controldim': 10,
    'lr': 0.001,
    'bs': 32,
    'epochs': 10,
}