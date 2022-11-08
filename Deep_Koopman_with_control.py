from config import *
from models import *
from utils import *
import joblib


print('Preparing training data')
training_loader, testing_loader, eval_data, (Xs_scaler, w_scaler, theta_scaler) = prepare_ncmapss_data(
    path=ncmapss_path,
    regime=regime,
    bs=model_params['bs'],
    train_units=train_units,
    test_units=test_units,
    cycles_frac=cycles_frac,
    eval_unit=eval_unit,
    eval_cycle=eval_cycle,
)
joblib.dump(Xs_scaler, 'models/Xs_scaler.save')
joblib.dump(w_scaler, 'models/w_scaler.save')
joblib.dump(theta_scaler, 'models/theta_scaler.save')



print('Initializing a model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepKoopmanControl(model_params).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'], weight_decay=1e-7)

# print('recovery weights', model.recovery.weight)
# print('shape', model.recovery.weight.shape)
# print('inverse', torch.inverse(model.recovery.weight))
# print('inverse shape', torch.inverse(model.recovery.weight).shape)

print('Training the model')
for epoch in range(1, model_params['epochs']+1):
    train_loss = train_model(model, optimizer, training_loader)
    print(f'Epoch {epoch}. Train loss {train_loss}')

    if (epoch % 5 == 0 or epoch == 1):
        with torch.no_grad():
            test_loss = test_model(model, testing_loader)
            print(f'Epoch {epoch}. Test loss {test_loss}')

    if epoch % 10 == 0:
        with torch.no_grad():
            forward_evaluation(model, eval_data)

    torch.save(model.state_dict(), f'models/model_epoch_{epoch}.pth')