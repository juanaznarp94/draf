import torch
from federated_ex.task import (
    model,
    load_data,
    get_weights,
    set_weights,
    train,
    validate_model,
    predict,
)
from torch.utils.data import DataLoader, TensorDataset


def test_model_output_shape():
    net = model()
    net.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = net(dummy_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"


def test_get_set_weights():
    net1 = model()
    net2 = model()
    weights = get_weights(net1)
    set_weights(net2, weights)
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        assert torch.allclose(p1, p2), "Parameters mismatch after setting weights"


def test_train_function_runs():
    net = model()
    trainloader,_ = load_data(1, 1000)
    loss = train(net, trainloader, epochs=1, device="cpu")
    assert isinstance(loss, float)
    assert loss > 0


def test_eval_function_runs():
    net = model()
    _,testloader = load_data(2, 1000)
    loss, acc =validate_model(net, testloader, device="cpu")
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 1


def test_predict_function_runs():
    net = model()
    _,testloader = load_data(90, 1000)
    predictions = predict(net, testloader, device="cpu")
    assert isinstance(predictions, list)
    assert all(isinstance(p, int) for p in predictions)
