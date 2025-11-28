from src.utils import gather_settings
from src.bin import Trainer, Tester


def main():
    print('Gathering settings...')
    settings = gather_settings()
    print('Setting up trainer...')
    trainer = Trainer(settings)
    print('Running train...')
    trainer.train()
    print('Setting up tester...')
    tester = Tester(settings)
    print('Running test...')
    tester.test()
    print('Done!')


if __name__ == '__main__':
    main()