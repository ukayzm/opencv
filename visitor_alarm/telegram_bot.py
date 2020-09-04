#!/usr/bin/env python3

from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
import logging

class TelegramBot():
    def __init__(self, token, chat_id=None):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

        self.token = token
        self.chat_id = chat_id
        self.updater = Updater(token=token, use_context=True)
        dispatcher = self.updater.dispatcher

        handler = CommandHandler('caps', self.caps)
        dispatcher.add_handler(handler)

        handler = CommandHandler('start', self.start)
        dispatcher.add_handler(handler)

        handler = MessageHandler(Filters.text & (~Filters.command), self.echo)
        dispatcher.add_handler(handler)

        # unknown handler should be added last
        handler = MessageHandler(Filters.command, self.unknown)
        dispatcher.add_handler(handler)

    def set_chat_id(self, chat_id):
        self.chat_id = chat_id

    def start_polling(self):
        self.updater.start_polling()

    def stop(self):
        self.updater.stop()

    def idle(self):
        self.updater.idle()

    def start(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")

    def caps(self, update, context):
        text_caps = ' '.join(context.args).upper()
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=text_caps)

    def capture(self, update, context):
        text_caps = ' '.join(context.args).upper()
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=text_caps)

    def echo(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=update.message.text)

    def unknown(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="Sorry, I didn't understand that command.")


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="Telegram Bot Token")
    args = ap.parse_args()

    bot = TelegramBot(args.token)
    bot.start_polling()
    print("telegram bot with token", args.token)
    print("press ^C to stop...")
    bot.idle()
    print("telegram bot finished")

