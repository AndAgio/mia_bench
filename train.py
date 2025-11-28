from src.utils import gather_settings
from src.bin import Trainer


def main():
    print('Gathering settings...')
    settings = gather_settings()
    print('Setting up trainer...')
    trainer = Trainer(settings)
    print('Running train...')
    trainer.train()
    print('Done!')


if __name__ == '__main__':
    main()