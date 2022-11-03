from config import *
from models import *
from utils import *


print('Preparing training data')
training_loader, testing_loader, (Xs_normalizer, w_normalizer, theta_normalizer) = prepare_ncmapss_data(
    path=ncmapss_path,
    regime=regime,
    bs=model_params['bs'],
    train_units=train_units,
    test_units=test_units,
    cycles_frac=cycles_frac,
)

print('Initializing a model')
model = DeepKoopmanControl(model_params)
optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'], weight_decay=1e-7)

print('Training the model')
for epoch in range(1, model_params['epochs']+1):
    train_loss = train_model(model, optimizer, training_loader)
    print(f'Epoch {epoch}. Train loss {train_loss}')

    if (epoch % 5 == 0 or epoch == 1):
        with torch.no_grad():
            test_loss = test_model(model, testing_loader)
            print(f'Epoch {epoch}. Test loss {test_loss}')