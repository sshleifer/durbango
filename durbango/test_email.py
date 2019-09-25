

from .send_gmail import send_gmail, HOSTNAME
import unittest


class TestUtils(unittest.TestCase):

    def test_gmail(self):
        send_gmail(f'Testing Auth from {HOSTNAME}', dry=False)
