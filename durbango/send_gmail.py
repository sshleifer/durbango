"""THIS DOESNT WORK ON REMOTE MACHINES. USE send_sms instead"""

import os
import smtplib
import subprocess

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#from email.mime.base import MIMEBase
#from email import encoders

EXAMPLE_BODY = '''<h4></h4><style type='text/css'>     table, tr, td{         border-collapse: collapse;         text-align: left;     }      caption {         font-weight: bold;         margin-bottom: 5px;     }      thead th, thead td{         background: #cdc2b1 none repeat scroll 0 0;         border-color: white;         border-style: solid;         border-width: 1pt;         height: 13.45pt;         padding: 0px 5px;         white-space: nowrap;         text-align: left;     }      thead tr td p{         margin-left: 5.75pt;         font-size: 12pt;         font-family: 'Calibri','sans-serif';     }      tbody .index {         text-align: left;     }      tbody tr{         height: 13.45pt;     }      tbody td, tbody th{         height: 13.45pt;         background: #ece6dd none repeat scroll 0 0;         border-color: white;         border-style: solid;         border-width: 1pt;         padding: 0px 5px;     }      tbody .text {         text-align: left;         font-family: 'Times New Roman', '

 serif';         font-size: 12pt;         margin: 0 0 0.0001pt;     }      tbody .positive {         background-color: #78E65C;     }      tbody .negative {         background-color: #E65C73;     }      table {         float: left;         margin: 4px;         margin-top: 8px;     } </style><table border='1'><thead><caption>Today Net Buying (predicted $MM 10yr equivalents)</caption><tr><th class='header nr_0 name_security'>Security</th><th class='header nr_1 name_ASSET MANAGER'>Asset Manager</th><th class='header nr_2 name_PENSION'>Pension</th><th class='header nr_3 name_HEDGE FUND'>Hedge Fund</th><th class='header nr_4 name_Total'>Total</th></tr></thead><tbody><tr><td class='nr_0 text'>2-3yr</td><td class='nr_1'>42</td><td class='nr_2'>11</td><td class='nr_3'>-23</td><td class='nr_4'>30</td></tr><tr><td class='nr_0 text'>5yr</td><td class='nr_1'>-14</td><td class='nr_2'>12</td><td class='nr_3'>-26</td><td class='nr_4'>-28</td></tr><tr><td class='nr_0 text'>7yr</td><td class='nr_1'>3

 3</td><td class='nr_2'>12</td><td class='nr_3'>2</td><td class='nr_4'>47</td></tr><tr><td class='nr_0 text'>10yr</td><td class='nr_1'>54</td><td class='nr_2'>-12</td><td class='nr_3'>96</td><td class='nr_4'>138</td></tr><tr><td class='nr_0 text'>30yr</td><td class='nr_1'>227</td><td class='nr_2'>31</td><td class='nr_3'>44</td><td class='nr_4'>302</td></tr><tr><td class='nr_0 text'>Total</td><td class='nr_1'>342</td><td class='nr_2'>54</td><td class='nr_3'>93</td><td class='nr_4'>489</td></tr></tbody></table><h4></h4><style type='text/css'>     table, tr, td{         border-collapse: collapse;         text-align: left;     }      caption {         font-weight: bold;         margin-bottom: 5px;     }      thead th, thead td{         background: #cdc2b1 none repeat scroll 0 0;         border-color: white;         border-style: solid;         border-width: 1pt;         height: 13.45pt;         padding: 0px 5px;         white-space: nowrap;         text-align: left;     }      thead tr t

 d p{         margin-left: 5.75pt;         font-size: 12pt;         font-family: 'Calibri','sans-serif';     }      tbody .index {         text-align: left;     }      tbody tr{         height: 13.45pt;     }      tbody td, tbody th{         height: 13.45pt;         background: #ece6dd none repeat scroll 0 0;         border-color: white;         border-style: solid;         border-width: 1pt;         padding: 0px 5px;     }      tbody .text {         text-align: left;         font-family: 'Times New Roman', 'serif';         font-size: 12pt;         margin: 0 0 0.0001pt;     }      tbody .positive {         background-color: #78E65C;     }      tbody .negative {         background-color: #E65C73;     }      table {         float: left;         margin: 4px;         margin-top: 8px;     } </style><table border='1'><thead><caption>Yesterday Net Buying ($MM 10yr equivalents)</caption><tr><th class='header nr_0 name_security'>Security</th><th class='header nr_1 name_ASSET MANAGER'>Asset Manager</th><th class='header nr_2 name_PENSION'>Pension</th><th class='header nr_3 name_HEDGE FUND'>Hedge Fund</th><th class='header nr_4 name_Total'>Total</th></tr></thead><tbody><tr><td class='nr_0 text'>2-3yr</td><td class='nr_1 negative'>-6</td><td class='nr_2 negative'>0</td><td class='nr_3 positive'>204</td><td class='nr_4'>198</td></tr><tr><td class='nr_0 text'>5yr</td><td class='nr_1 negative'>-86</td><td class='nr_2 negative'>0</td><td class='nr_3 positive'>552</td><td class='nr_4'>466</td></tr><tr><td class='nr_0 text'>7yr</td><td class='nr_1 negative'>-195</td><td class='nr_2 negative'>-32</td><td class='nr_3 negative'>-110</td><td class='nr_4'>-337</td></tr><tr><td class='nr_0 text'>10yr</td><td class='nr_1 negative'>-113</td><td class='nr_2 positive'>113</td><td class='nr_3'>182</td><td class='nr_4'>182</td></tr><tr><td class='nr_0 text'>30yr</td><td class='nr_1 negative'>-28</td><td class='nr_2'>23</td><td class='nr_3 negative'>-553</td><td class='nr_4'>-558</td></tr><t

 r><td class='nr_0 text'>Total</td><td class='nr_1'>-428</td><td class='nr_2'>104</td><td class='nr_3'>275</td><td class='nr_4'>-49</td></tr></tbody></table>
'''.replace('\n', '')

HOSTNAME = subprocess.getoutput('hostname')
def send_gmail(subject: str, html_body=None, toaddr=None, fromaddr=None,
               dry=False, pw=None):
    if html_body is None: html_body = ''
    if fromaddr is None:
        fromaddr = os.getenv('GMAIL')
    if toaddr is None:
        toaddr = fromaddr  # Default send to self
    if pw is None:
        pw = os.getenv('GMAIL_PW')
    if isinstance(toaddr, str):
        to_addr = [toaddr]
    # = listify(toaddr)

    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = ', '.join(toaddr)
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))
    if dry:
        print(f'Dry mode: Qutting before hitting gmail server.')
        return

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    #server.set_debuglevel(1)
    server.login(fromaddr, pw)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

if __name__ == '__main__':
    send_gmail(f'Testing Auth from {HOSTNAME}', dry=False)
