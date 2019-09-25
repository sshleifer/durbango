import os


# http://forums.fast.ai/t/send-yourself-a-text-when-training-is-complete/5256
# pip install twilio
from twilio.rest import Client

TWILIO_TOK, TWILIO_ID, TWILIO_NUM, RECIPIENT_NUM = (
    os.environ['TWILIO_TOK'], os.environ['TWILIO_ID'],
    os.environ['TWILIO_NUM'], os.environ['TWILIO_RECIPIENT_NUM']
)

def send_sms(message: str, to=RECIPIENT_NUM, tok=TWILIO_TOK, id=TWILIO_ID, twilio_num=RECIPIENT_NUM):
    """Send a text message after long job finishes. Texts occasionally send up to 20 minutes late.
    Args:
        message: body of message
        to: defaults to $TWILIO_RECIPIENT_NUM
        tok: twilio token, defaults to $TWILIO_TOK
        id: $TWILIO_ID
        twilio_num: $TWILIO_NUM (text will be sent from this)
    """
    client = Client(id, tok)
    client.messages.create(from_=twilio_num, to=to, body=message)
    print(f'Sent text to {RECIPIENT_NUM}')


if __name__ == '__main__':
    send_sms('testing from command line')
