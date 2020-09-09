# Visitor Alarm

This is a Telegram Bot doing the folowings:

* read video from web cam
* detect faces in the frame
* compare the face with the previously saved faces
* if new person is detected, send a message via Telegram

This code is highly refering to [unknown_face_classifier](https://github.com/ukayzm/opencv/tree/master/unknown_face_classifier). I recommend you to read it first.

# Usage

```bash
$ python visitor_alarm.py -h
usage: visitor_alarm.py [-h] --token TOKEN [--srcfile SRCFILE]

optional arguments:
  -h, --help         show this help message and exit
  --token TOKEN      Telegram Bot Token
  --srcfile SRCFILE  Video file to process. If not specified, web cam is used.
```

You have to make a Telegram bot before doing this. Please search Google how to make a Telegram Bot.

Once you make the bot, pass the token as a parameter like this.

```bash
python visitor_alarm.py --token '1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHI'
```

