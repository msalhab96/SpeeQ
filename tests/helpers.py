IGNORE_USERWARNING = 'ignore::UserWarning'


def check_grad(result, model):
    """Checks the model's grads
    """
    loss = result.mean()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None
