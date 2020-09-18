# Visitor Alarm Telegram

This is a Telegram Bot doing the folowings:

* read video from web cam (or a video file)
* detect faces in the picture
* compare the face with the previously saved faces
* if new person is detected or the person appears again, send a message via Telegram

This code is highly refering to [unknown_face_classifier](https://github.com/ukayzm/opencv/tree/master/unknown_face_classifier). I recommend you to read it first.

# Usage

```bash
$ python visitor_alarm_telegram.py -h
usage: visitor_alarm_telegram.py [-h] --token TOKEN [--srcfile SRCFILE]
                                 [--threshold THRESHOLD] [--sbf SBF]
                                 [--resize-ratio RESIZE_RATIO]
                                 [--appearance-interval APPEARANCE_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --token TOKEN         Telegram Bot Token
  --srcfile SRCFILE     Video file to process. If not specified, web cam is
                        used.
  --threshold THRESHOLD
                        threshold of the similarity (default=0.42)
  --sbf SBF             second between frame processed (default=0.5)
  --resize-ratio RESIZE_RATIO
                        resize the frame to process (less time, less accuracy)
  --appearance-interval APPEARANCE_INTERVAL
                        alarm interval second between appearance (default=10)
```

You have to make a Telegram bot before doing this. Please search Google for how to make a Telegram Bot.

Once you make the bot, execute visitor_alarm_telegram.py with the token as a parameter.

```bash
$ python visitor_alarm_telegram.py --token '1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHI'
Visitor Alarm Telegram is started.
* srcfile = 0 (webcam)
* resize_ratio = 1.0
* sbf (second between frame processed) = 0.5
* similarity threshold = 0.42
* appearance_interval = 10
press ^C to stop...
```

Or you can specify `--srcfile` parameter to use a video file instead of webcam. This is useful for the testing purpose.

```bash
$ python visitor_alarm_telegram.py --token '1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHI' --srcfile ~/Videos/extj.mp4
```

Then, open a chat with the bot in Telegram app on your phone. Input `/start` to start face classifier. Input `/help` for more commands.

| Available Commands | Comments |
|--------------------|----------|
| /help | show available commands |
| /settings | show settings |
| /start | start face classifier |
| /stop | stop face classifier |
| /status | show the status of person DB and face classifier |
| /shot | show the current screen of web cam (or the video) |
| /rename old_name new_name | change the person's name |
| /list | list all persons with picture |

# Screen shots

## settings

Show settings.

<p align="center">
   <img src="png/vat_settings.png">
</p>

## start

Start face classifier.

<p align="center">
   <img src="png/vat_start.png">
</p>

## On a person appeared first

When new person is detected, a message is delivered to the user.

<p align="center">
   <img src="png/vat_person_first.png">
</p>

## On the person appeared again

When the person is detected again, a message is delivered to the user.

<p align="center">
   <img src="png/vat_person_again.png">
</p>

## status

Show the current status.

<p align="center">
   <img src="png/vat_status.png">
</p>

## stop

Stop face classifier.

<p align="center">
   <img src="png/vat_stop.png">
</p>

## shot

You can see the screen shot of the current video.

<p align="center">
   <img src="png/vat_shot.png">
</p>

## rename

You can change the name of person.

<p align="center">
   <img src="png/vat_rename.png">
</p>

## list

You can check the list of persons.

<p align="center">
   <img src="png/vat_list.png">
</p>

