#!/usr/bin/env python3

from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from telegram.error import (TelegramError, Unauthorized, BadRequest, 
                            TimedOut, ChatMigrated, NetworkError)
import logging
import person_db
import face_classifier
import io
import cv2


class CmdDefault():
    def __init__(self, name=None, face_classifier=None, person_db=None):
        if name is None:
            name = self.__class__.__name__[3:].lower()
        self.name = name
        self.fc = face_classifier
        self.pdb = person_db

    def usage(self):
        return '/' + self.name

class CmdName(CmdDefault):
    def usage(self):
        return '/' + self.name + ' old_name new_name'

    def method(self, update, context):
        org_msg = update.message.text
        args = org_msg.split()
        usage = 'usage: /name old_name new_name'
        if len(args) == 3:
            if self.pdb.rename(args[1], args[2]) == 0:
                reply = 'name changed ' + args[1] + ' -> ' + args[2]
            else:
                reply = 'cannot change the person with name' + args[1]
        else:
            reply = self.usage()
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=reply)

class CmdList(CmdDefault):
    def method(self, update, context):
        chat_id = update.effective_chat.id
        for person in self.pdb.persons:
            reply = "%s with %d faces" % (person.name, len(person.faces))
            face = person.faces[0]
            is_success, buf = cv2.imencode(".jpg", face.image)
            bio = io.BytesIO(buf)
            bio.seek(0)
            context.bot.send_photo(chat_id=chat_id, photo=bio,
                                   caption=reply)

class CmdStatus(CmdDefault):
    def method(self, update, context):
        reply = self.pdb.__repr__()
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=reply)
        reply = " ".join(context.args)
        update.message.reply_text("context.args: " + reply)


class TelegramBot():
    def add_command(self, handlerObj):
        self.usages.append(handlerObj.usage())
        handler = CommandHandler(handlerObj.name, handlerObj.method)
        self.updater.dispatcher.add_handler(handler)

    def __init__(self, token, chat_id=None, face_classifier=None,
                 person_db=None):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

        self.token = token
        self.chat_id = chat_id
        self.updater = Updater(token=token, use_context=True)

        self.fc = face_classifier
        self.pdb = person_db

        self.handlers = []
        self.usages = []
        self.add_command(CmdName(person_db=self.pdb))
        self.add_command(CmdList(person_db=self.pdb))
        self.add_command(CmdStatus(person_db=self.pdb))

        # help command handler
        handler = CommandHandler('help', self.help)
        self.updater.dispatcher.add_handler(handler)

        # unknown handler should be added last
        handler = MessageHandler(Filters.command, self.unknown)
        self.updater.dispatcher.add_handler(handler)

        # error handler
        self.updater.dispatcher.add_error_handler(self.error_callback)

    # error handler
    # https://github.com/python-telegram-bot/python-telegram-bot/wiki/Exception-Handling
    def error_callback(self, update, context):
        try:
            raise context.error
        except Unauthorized:
            # remove update.message.chat_id from conversation list
            print("Unauthorized error")
        except BadRequest:
            # handle malformed requests
            print("BadRequest error")
        except TimedOut:
            # handle slow connection problems
            print("TimedOut error")
        except NetworkError:
            # handle other connection problems
            print("NetworkError error")
        except ChatMigrated as e:
            # the chat_id of a group has changed, use e.new_chat_id instead
            print("ChatMigrated error")
        except TelegramError:
            # handle all other telegram related errors
            print("TelegramError error")

    def start_polling(self):
        self.updater.start_polling()

    def stop(self):
        self.updater.stop()

    def idle(self):
        self.updater.idle()

    def unknown(self, update, context):
        reply = '\n'.join(["Sorry, I didn't understand that command.",
                           "Try /help for available commands"])
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=reply)

    def help(self, update, context):
        reply = '\n'.join(self.usages)
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=reply)




if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="Telegram Bot Token")
    args = ap.parse_args()

    dir_name = "result"
    pdb = person_db.PersonDB()
    pdb.load_db()

    bot = TelegramBot(args.token, person_db=pdb)
    bot.start_polling()
    print("telegram bot with token", args.token)
    print("press ^C to stop...")
    bot.idle()
    print("telegram bot finished")

