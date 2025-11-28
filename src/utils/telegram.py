import telegram
import pathlib
import yaml
import os

from .variables import PATH_REPO

class TelBotManager():
    def __init__(self, secrets_file = None):
        if secrets_file is None:
            secrets_file = os.path.join(PATH_REPO, "secrets.yml")
        self.get_bot(secrets_file)

    def get_bot(self, secrets_file):
        with open(secrets_file, "r") as readfile:
            secrets = yaml.safe_load(readfile)
        self.telegram_bot = telegram.Bot(token=secrets['telegram']['bot'])
        self.telegram_channel = secrets['telegram']['channel']

    async def send_message(self, text):
        async with self.telegram_bot:
            await self.telegram_bot.send_message(text=text,
                                                chat_id=self.telegram_channel)

    async def run_bot(self, messages):
        text = '\n'.join(messages)
        await self.send_message(text)

