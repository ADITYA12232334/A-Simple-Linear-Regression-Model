from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

weights=0.3
bias=0.9
start=0
end=2
step=0.01
X=torch.arange(start,end,step).unsqueeze(dim=1)
y=weights*X + bias
training_part=int(0.8*len(X))
X_train,y_train=X[:training_part],y[:training_part]
X_test,y_test=X[training_part:],y[training_part:]
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
        plt.figure(figsize=(10,7))
        plt.scatter(train_data,train_labels,c='g',s=4,label="train_data")
        plt.scatter(test_data,test_labels,c='b',s=4,label="testing data")
        if predictions is not None:
             plt.scatter(test_data,predictions,c='r',s=4,label="model predictions")
        plt.legend(prop={"size":14})
        plt.show()  
plot_predictions()   
class Linearregressionmodelv3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear_layer=nn.Linear(in_features=1,
                                    out_features=1)
    def forward(self,x:torch.tensor)->torch.tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_2=Linearregressionmodelv3()
model_2=model_2.to(device)
X_train=X_train.to(device)
y_train=y_train.to(device)
X_test=X_test.to(device)
y_test=y_test.to(device)
print(model_2.state_dict())
loss_fn=nn.L1Loss()
optimizer=torch.optim.SGD(params=model_2.parameters(),lr=0.01)

epochs=300
for epoch in range(epochs):
    model_2.train()
    train_preds=model_2(X_train)
    train_loss=loss_fn(train_preds,y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_preds=model_2(X_test)
        test_loss=loss_fn(test_preds,y_test)
    if epoch%20==0:
        print(f"epoch: {epoch} | loss: {train_loss} | test loss:{test_loss}")
print(model_2.state_dict())

with torch.inference_mode():
     y_preds=model_2(X_test)
     y_preds=y_preds.cpu()
     plot_predictions(predictions=y_preds)

from pathlib import Path
MODEL_PATH=Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)
MODEL_NAME="execise_2_model.pth"
MODEL_SAVE_PATH=MODEL_PATH / MODEL_NAME
print(f"saving to :{MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(),f=MODEL_SAVE_PATH)

model_3=Linearregressionmodelv3()
model_3.load_state_dict(torch.load(MODEL_SAVE_PATH))