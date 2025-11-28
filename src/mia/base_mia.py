

class MIA():
    # TODO: Implement attacker functionalities.
    # Issue URL: https://github.com/AndAgio/mia_bench/issues/1
    # assignee: AndAgio
    def __init__(self, victim_model, dataset, n_auditing_samples):
        self.victim_model = victim_model
        self.dataset = dataset
        self.n_auditing_samples = n_auditing_samples
