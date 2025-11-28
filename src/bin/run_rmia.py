


def main():
    
    victim = Victim(settings.dataset, settings.victim_model)
    victim.train_model()

    attacker = Mia()
    auditing_dataset = attacker.build_auditing_dataset(victim_dataset)
    attacker.optimize()
    attacker.measure_effectiveness(auditing_dataset)