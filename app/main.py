from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import quantile_cpp  
import json

app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
# Modelo simple de regresión
class QuantileRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(QuantileRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# Función de pérdida cuantil (usando C++)
def quantile_loss(y_true, y_pred, quantile):
    loss = quantile_cpp.quantile_loss(y_true.numpy(), y_pred.detach().numpy(), quantile)
    return torch.tensor(loss.mean(), requires_grad=True)

@app.post("/quantile-loss")
def calculo():
    
    # Generar datos de ejemplo
    torch.manual_seed(0)
    X = torch.rand(100, 1)
    y = 3 * X.squeeze() + torch.randn(100) * 0.5

    # Preparar los datos para PyTorch
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Configuración del modelo
    model = QuantileRegressionModel(input_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Entrenamiento
    quantile = 0.5  # Mediana
    epochs = 100

    sepochs = ""
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X).squeeze()
            loss = quantile_loss(batch_y, y_pred, quantile)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            sepochs += f"Epoch {epoch}, Loss: {loss.item()}"

    # Evaluar el modelo
    with torch.no_grad():
        y_pred = model(X).squeeze()
        sepochs += f"Predicciones finales: {y_pred[:5]}"

    j1 = {
        "output": sepochs
    }
    jj = json.dumps(str(j1))

    return jj



    