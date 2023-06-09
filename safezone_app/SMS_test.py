import requests
import sys
import os
import hashlib
import hmac
import base64
import time, json

timestamp = int(time.time() * 1000)
timestamp = str(timestamp)

access_key = "oDgbh0ZJsFZJU4B02NSP"


url = "https://sens.apigw.ntruss.com"
uri = "/sms/v2/services/ncp:sms:kr:308300814920:safezone/messages"

def make_signature():
    secret_key = "kcdWYdxcH1Y0FwwzIl980FLz6ZDTV2DH15fivP2Z"
    secret_key = bytes(secret_key, 'UTF-8')
    method = "POST"
    message = method + " " + uri + "\n" + timestamp + "\n" + access_key
    message = bytes(message, 'UTF-8')
    signinkey_val = hmac.new(secret_key, message, digestmod=hashlib.sha256).digest()
    return base64.b64encode(signinkey_val).decode('UTF-8')

header = {
    'Content-Type' : 'application/json; charset=utf-8',
    'x-ncp-apigw-timestamp' : timestamp,
    'x-ncp-iam-access-key' : access_key,
    'x-ncp-apigw-signature-v2' : make_signature()
}

data = {
    "type" : "SMS",
    "from" : "01097069798",
    "subject" : "발신번호테스트",
    "content" : "문자내용테스트",
    "messages" :[{"to" : "01083345690"}]
    
}

res = requests.post(url+uri,headers=header,data=json.dumps(data))
print(res.json())