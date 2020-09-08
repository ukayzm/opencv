#!/usr/bin/env python3

import telegram
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
    def __init__(self, telegram_bot, name=None):
        if name is None:
            name = self.__class__.__name__[3:].lower()
        self.name = name
        self.tb = telegram_bot

    def usage(self):
        return '/' + self.name

    def method(self, update, context):
        # dummy method - echo
        chat_id = update.effective_chat.id
        reply = update.message.text
        context.bot.send_message(chat_id=chat_id, text=reply)


class CmdName(CmdDefault):
    def usage(self):
        return '/' + self.name + ' old_name new_name'

    def method(self, update, context):
        chat_id = update.effective_chat.id
        args = update.message.text.split()
        if len(args) == 3:
            if self.tb.pdb.rename(args[1], args[2]) == 0:
                reply = 'name changed ' + args[1] + ' -> ' + args[2]
            else:
                reply = 'cannot change the person with name' + args[1]
        else:
            reply = 'usage: ' + self.usage()
        context.bot.send_message(chat_id=chat_id, text=reply)

class CmdList(CmdDefault):
    def method(self, update, context):
        chat_id = update.effective_chat.id
        for person in self.tb.pdb.persons:
            reply = "%s with %d faces" % (person.name, len(person.faces))
            face = person.faces[0]
            is_success, buf = cv2.imencode(".jpg", face.image)
            bio = io.BytesIO(buf)
            bio.seek(0)
            context.bot.send_photo(chat_id=chat_id, photo=bio,
                                   caption=reply)

class CmdStatus(CmdDefault):
    def method(self, update, context):
        chat_id = update.effective_chat.id
        reply = self.tb.pdb.__repr__()
        reply += '\n' + self.tb.fc.status_string
        context.bot.send_message(chat_id=chat_id, text=reply)
        reply = " ".join(context.args)
        update.message.reply_text("context.args: " + reply)

class CmdAlarm(CmdDefault):
    def usage(self):
        return '/' + self.name + ' [start|stop]'

    def method(self, update, context):
        chat_id = update.effective_chat.id
        args = update.message.text.split()
        if len(args) <= 1:
            if self.tb.alarm_receiver is None:
                reply = 'Visitor alarm is not enabled.'
            else:
                reply = 'Visitor alarm receiver is '
                if chat_id == self.tb.alarm_receiver:
                    reply += 'you.'
                else:
                    reply += str(self.tb.alarm_receiver) + '.'
        else:
            if args[1] == 'start':
                self.tb.alarm_receiver = chat_id
                reply = 'OK. From now visitor alarm will sent to you.'
                print('visitor alarm will be sent to ' + str(chat_id))
            elif args[1] == 'stop':
                self.tb.alarm_receiver = None
                reply = 'OK. From now visitor alarm will not be sent.'
                print('visitor alarm is disabled')
            else:
                reply = 'usage: ' + self.usage()
        context.bot.send_message(chat_id=chat_id, text=reply)

class CmdStart(CmdDefault):
    def usage(self):
        return '/' + self.name + ' [video_file]'

    def method(self, update, context):
        chat_id = update.effective_chat.id
        reply = self.tb.fc.settings.__repr__()
        if self.tb.fc.running:
            reply = 'Face classifier is already running.'
            reply += '\nStop first by /stop command.'
        else:
            reply = 'Start face classifier with source '
            reply += self.tb.fc.settings.source_file
            self.tb.fc.start_running()
            self.tb.alarm_receiver = chat_id
            reply += 'Visitor alarm will sent to you.'
            print('visitor alarm will be sent to ' + str(chat_id))
        context.bot.send_message(chat_id=chat_id, text=reply)

class CmdStop(CmdDefault):
    def method(self, update, context):
        chat_id = update.effective_chat.id
        reply = self.tb.fc.settings.__repr__()
        if self.tb.fc.running:
            self.tb.alarm_receiver = None
            reply = 'Stop'
            self.tb.fc.stop_running()
        else:
            reply = 'Face classifier is not running.'
        context.bot.send_message(chat_id=chat_id, text=reply)

class CmdSettings(CmdDefault):
    def method(self, update, context):
        chat_id = update.effective_chat.id
        reply = self.tb.fc.settings.__repr__()
        context.bot.send_message(chat_id=chat_id, text=reply)


class TelegramBot():
    def __init__(self, token, face_classifier=None, person_db=None):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

        self.token = token
        self.core = telegram.Bot(token)
        self.updater = Updater(token=token, use_context=True)

        self.fc = face_classifier
        self.fc.set_on_new_person(self.on_new_person)
        self.pdb = person_db
        self.alarm_receiver = None

        self.usages = []
        self.add_command(CmdName(self))
        self.add_command(CmdList(self))
        self.add_command(CmdStatus(self))
        self.add_command(CmdAlarm(self))
        self.add_command(CmdStart(self))
        self.add_command(CmdStop(self))
        self.add_command(CmdSettings(self))

        # help command handler
        handler = CommandHandler('help', self.help)
        self.updater.dispatcher.add_handler(handler)

        # unknown handler should be added last
        handler = MessageHandler(Filters.command, self.unknown)
        self.updater.dispatcher.add_handler(handler)

        # error handler
        self.updater.dispatcher.add_error_handler(self.error_callback)

    def add_command(self, handlerObj):
        self.usages.append(handlerObj.usage())
        handler = CommandHandler(handlerObj.name, handlerObj.method)
        self.updater.dispatcher.add_handler(handler)

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
        chat_id = update.effective_chat.id
        reply = '\n'.join(["Sorry, I didn't understand that command.",
                           "Try /help for available commands"])
        context.bot.send_message(chat_id=chat_id, text=reply)

    def help(self, update, context):
        chat_id = update.effective_chat.id
        reply = '\n'.join(self.usages)
        context.bot.send_message(chat_id=chat_id, text=reply)

    def on_new_person(self, person):
        print('on_new_person', person)
        chat_id = self.alarm_receiver
        reply = "new person %s with %d faces" % (person.name, len(person.faces))
        face = person.faces[0]
        is_success, buf = cv2.imencode(".jpg", face.image)
        bio = io.BytesIO(buf)
        bio.seek(0)
        self.core.send_photo(chat_id=chat_id, photo=bio,
                             caption=reply)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="Telegram Bot Token")
    args = ap.parse_args()

    dir_name = "result"
    pdb = person_db.PersonDB()
    pdb.load_db()

    fc = face_classifier.FaceClassifier(pdb)

    bot = TelegramBot(args.token, face_classifier=fc, person_db=pdb)
    bot.start_polling()
    print("telegram bot with token", args.token)
    print("press ^C to stop...")
    bot.idle()
    fc.stop_running()
    print("telegram bot finished")

