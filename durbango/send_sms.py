import os


# http://forums.fast.ai/t/send-yourself-a-text-when-training-is-complete/5256
# pip install twilio
from twilio.rest import Client

TWILIO_TOK, MY_NUM, TWILIO_ID = os.environ['TWILIO_TOK'], os.environ['MY_NUM'], os.environ['TWILIO_ID']

def send_sms(
        message, to=MY_NUM, tok=TWILIO_TOK, id=TWILIO_ID, twilio_num='+15085440501'):
    """Send a text message """
    client = Client(id, tok)
    client.messages.create(from_=twilio_num, to=to, body=message)
    print(f'Sent text to {MY_NUM}')


if __name__ == '__main__':
    send_sms('testing from command line')
