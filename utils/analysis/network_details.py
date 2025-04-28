def print_aux_weights(aux_agent, n=None):
    for name, param in aux_agent.policy.named_parameters():
        weights = param.detach().cpu().view(-1)
        to_print = weights if n is None else weights[:n]
        print(f"{name}: {to_print}")