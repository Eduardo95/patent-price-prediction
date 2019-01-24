#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from PIL import Image
import hashlib
import re
import random
import operator
import urllib.request
from datetime import *


class APIClient(object):
    def http_request(self, url, paramDict):
        # post_content = ''
        # for key in paramDict:
        #     post_content = post_content + '%s=%s&' % (key, paramDict[key])
        # post_content = post_content[0:-1]
        post_content = urllib.parse.urlencode(paramDict).encode('utf-8')
        # print(post_content)
        # data参数如果要传必须传bytes（字节流）类型的，如果是一个字典，先用urllib.parse.urlencode()编码
        req = urllib.request.Request(url, data=post_content)
        req.add_header('Content-Type', 'application/x-www-form-urlencoded')
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
        response = opener.open(req, post_content)
        return response.read()

    def http_upload_image(self, url, paramKeys, paramDict, filebytes):
        # timestr要先通过ascii编码encode才能hash
        timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode('ascii')
        boundary = '------------' + hashlib.md5(timestr).hexdigest().lower()
        boundarystr = '\r\n--%s\r\n' % (boundary)

        bs = b''
        for key in paramKeys:
            bs = bs + boundarystr.encode('ascii')
            param = "Content-Disposition: form-data; name=\"%s\"\r\n\r\n%s" % (key, paramDict[key])
            # print param
            bs = bs + param.encode('utf8')
        bs = bs + boundarystr.encode('ascii')

        header = 'Content-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\nContent-Type: image/gif\r\n\r\n' % (
            'sample')
        bs = bs + header.encode('utf8')

        bs = bs + filebytes
        tailer = '\r\n--%s--\r\n' % (boundary)
        bs = bs + tailer.encode('ascii')

        import requests
        headers = {'Content-Type': 'multipart/form-data; boundary=%s' % boundary,
                   'Connection': 'Keep-Alive',
                   'Expect': '100-continue',
                   }
        response = requests.post(url, params='', data=bs, headers=headers)
        return response.text


def arguments_to_dict(args):
    argDict = {}
    if args is None:
        return argDict

    count = len(args)
    if count <= 1:
        print('exit:need arguments.')
        return argDict

    for i in [1, count - 1]:
        pair = args[i].split('=')
        if len(pair) < 2:
            continue
        else:
            argDict[pair[0]] = pair[1]

    return argDict


if __name__ == '__main__':
    client = APIClient()
    while 1:
        paramDict = {}
        result = ''
        act = input('Action:')
        if operator.eq(act, 'info'):
            paramDict['username'] = input('username:')
            paramDict['password'] = input('password:')
            result = client.http_request('http://api.ruokuai.com/info.xml', paramDict)
        elif operator.eq(act, 'register'):
            paramDict['username'] = input('username:')
            paramDict['password'] = input('password:')
            paramDict['email'] = input('email:')
            result = client.http_request('http://api.ruokuai.com/register.xml', paramDict)
        elif operator.eq(act, 'recharge'):
            paramDict['username'] = input('username:')
            paramDict['id'] = input('id:')
            paramDict['password'] = input('password:')
            result = client.http_request('http://api.ruokuai.com/recharge.xml', paramDict)
        elif operator.eq(act, 'url'):
            paramDict['username'] = input('username:')
            paramDict['password'] = input('password:')
            paramDict['typeid'] = input('typeid:')
            paramDict['timeout'] = input('timeout:')
            paramDict['softid'] = input('softid:')
            paramDict['softkey'] = input('softkey:')
            paramDict['imageurl'] = input('imageurl:')
            result = client.http_request('http://api.ruokuai.com/create.xml', paramDict)
        elif operator.eq(act, 'report'):
            paramDict['username'] = input('username:')
            paramDict['password'] = input('password:')
            paramDict['id'] = input('id:')
            result = client.http_request('http://api.ruokuai.com/create.xml', paramDict)
        # 实际调用是这一部分的功能
        elif operator.eq(act, 'upload'):
            paramDict['username'] = input('username:')
            paramDict['password'] = input('password:')
            paramDict['typeid'] = input('typeid:')
            paramDict['timeout'] = input('timeout:')
            paramDict['softid'] = input('softid:')
            paramDict['softkey'] = input('softkey:')
            paramKeys = ['username',
                         'password',
                         'typeid',
                         'timeout',
                         'softid',
                         'softkey'
                         ]

            imagePath = input('Image Path:')
            img = Image.open(imagePath)
            if img is None:
                print('get file error!')
                continue
            img.save("upload.gif", format="gif")
            filebytes = open("upload.gif", "rb").read()
            result = client.http_upload_image("http://api.ruokuai.com/create.xml", paramKeys, paramDict, filebytes)

        elif operator.eq(act, 'help'):
            print('info')
            print('register')
            print('recharge')
            print('url')
            print('report')
            print('upload')
            print('help')
            print('exit')
        elif operator.eq(act, 'exit'):
            break

        # 返回计算结果
        pattern = '(<Result>)([0-9]+)(</Result>)'
        print(re.search(pattern, str(result)).group(2))
        # print(result)

# Rollingegg
# LWJsteve98
