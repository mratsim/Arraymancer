import json
import torch
import torch.optim as optim

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def drosenbrock(x, y):
    dx = torch.Tensor([-400 * x * (y - x ** 2) - 2 * (1 - x)])
    dy = torch.Tensor([200 * (y - x ** 2)])
    return (dx, dy)

# Print the configuration nicely instead of a disgusting dict.
def pretty_format(config):
    s = ""
    for key, value in config["param_groups"][0].items():
        if key != "params":
            s += f"{key}: {value}, "
    return s[:-1] # Strips the final comma

algorithms = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    #'adamw': optim.Adamw, # Bleeding Edge optimizer not in 1.1.0 but is in HEAD
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'sgd': optim.SGD,
}

with open('tests.json', 'r') as f:
    tests = json.loads(f.read())

for test in tests:
    print(test['algorithm'] + '\t')
    opt_type = algorithms[test['algorithm']]
    for config in test['config']:
        # Print the config so we know what this run is.
        print(pretty_format(config))
        print('================================================================================\t')
        params = (torch.Tensor([1.5]), torch.Tensor([1.5]))
        # As far as I'm aware a learning rate is always required.
        opt = opt_type(params, 1e-4)
        # To load the proper config we load state_dicts for the optimizer which
        # is the easiest way to load it from JSON since the non legacy optim
        # module redefined Optimizers to no longer take in dictionaries of
        # their parameter options.
        config["param_groups"][0]["params"] = params
        opt.load_state_dict(config)
        for epoch in range(1, 101):
            val = float(rosenbrock(params[0], params[1]))
            print(f"Epoch {epoch:>3}/100: ({float(params[0]):>.12f}, {float(params[1]):>.12f}) = {val}")

            dx, dy = drosenbrock(params[0], params[1])
            params[0].grad = dx
            params[1].grad = dy
            opt.step()

        print()

