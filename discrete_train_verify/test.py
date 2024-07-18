import torch

from models import LogisticRegressionModel
from torch.utils.data import DataLoader, TensorDataset
from utils import calculate_metrics


def test_model_mlp(model, image, true_label):
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            image = image.cuda()
        outputs = model(image)
        if isinstance(model, LogisticRegressionModel):
            predicted = model.predict(image)
            # predicted = predicted.cuda()
        else:
            _, predicted = torch.max(outputs, 1)
        total = image.size(0)
        correct = (predicted == true_label).sum().item()
        # FPR, FNR, ACC = calculate_metrics(predicted, true_label)
        # print(f"FPR {FPR*100:.3f} FNR {FNR*100:.3f} ACC {ACC*100:.3f}")
    return outputs, predicted, 1.0 * correct / total
