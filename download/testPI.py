import requests
import xlrd
import base64
from selenium.webdriver.common.keys import Keys
import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time
import os
import zipfile
import sys
from download import ruokuai # 若快打码平台，用以自动输入验证码
from PIL import Image
import re

# --------------固定信息包括文件路径-----------------
header = {
    "Accept": "text / html, * / *;q = 0.01",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36"
}

codeurl = 'http://www.pss-system.gov.cn/sipopublicsearch/portal/login-showPic.shtml' # 国家专利局的网址
targeturl = 'http://www.pss-system.gov.cn/sipopublicsearch/portal/uiIndex.shtml' # 登录成功的url
webdriver_path = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver" # webdriver浏览器插件下载的位置
file_path = '专利交易案例.xlsx'   # 专利交易案例表格文件
filename = r'd:\Downloads\downfile.zip'  # 要解压的文件，这是从chrome下载的文件路径
filedir = r'd:\\testData'  # 解压后放入的目录
# ------------------------------------------------------


'''将excel文件信息导出datalist'''
def from_excel(file):
    wb = xlrd.open_workbook(file)
    ws = wb.sheet_by_name('Sheet1')
    datalist = []

    for r in range(ws.nrows):
        col = []
        for c in range(ws.ncols):
            col.append(ws.cell(r, c).value)
        datalist.append(col)

    # print(datalist)
    return datalist


'''excel表格专利号的处理'''
def change_str(s):
    re1 = r'[0-9]+\.';
    name = re.findall(re1, s)
    if name:
        name[0] = name[0].strip('.')
        length = len(name[0])
        return 'CN' + name[0][:length - 1] + '.' + name[0][length - 1:length]
    else:
        return 'CN' + s


def get_auth_code(driver, codeElement):
    '''获取登陆验证码'''
    driver.maximize_window()  # 将浏览器最大化
    driver.save_screenshot('login.png')  # 截取登录页面
    imgSize = codeElement.size  # 获取验证码图片的大小
    imgLocation = codeElement.location  # 获取验证码元素坐标
    # range = (int(imgLocation['x']), int(imgLocation['y']), int(imgLocation['x'] + imgSize['width']),
    #           int(imgLocation['y'] + imgSize['height']))  # 计算验证码整体坐标
    range = (1094, 530, 1179, 557)  # 计算验证码整体坐标
    login = Image.open("login.png")
    frame4 = login.crop(range)  # 截取验证码图片
    frame4.save('valcode.png')


def get_download_code(driver):
    '''获取下载验证码'''
    driver.maximize_window()  # 将浏览器最大化
    driver.save_screenshot('login.png')  # 截取登录页面
    range = (1094, 530, 1179, 557)  # 计算验证码整体坐标(注意！这里的坐标是截屏之后获得的，切勿修改)
    login = Image.open("login.png")
    frame4 = login.crop(range)  # 截取验证码图片
    frame4.save('valcode.png')


def validation_result():
    client = ruokuai.APIClient()
    paramDict = {}
    ''' 以下参数请勿修改'''
    paramDict['username'] = 'Rollingegg'
    paramDict['password'] = 'LWJsteve98'
    paramDict['typeid'] = '7100'
    paramDict['timeout'] = '30'
    paramDict['softid'] = '115489'
    paramDict['softkey'] = 'b7369eb491a54c60bce17d2c6ba09e41'
    paramKeys = ['username',
                 'password',
                 'typeid',
                 'timeout',
                 'softid',
                 'softkey'
                 ]
    # 验证码图片的路径，请保存在与本文件同一文件夹下
    imagePath = 'valcode.png'
    img = Image.open(imagePath)
    if img is None:
        print('get file error!')
        return
    img.save("upload.gif", format="gif")
    filebytes = open("upload.gif", "rb").read()
    result = client.http_upload_image("http://api.ruokuai.com/create.xml", paramKeys, paramDict, filebytes)
    # print(result)
    # 返回计算结果
    pattern = '(<Result>)([0-9]+)(</Result>)'
    return str(re.search(pattern, str(result)).group(2))


'''模拟浏览器下载专利文件压缩包'''
def get_download(url, key):
    browser = webdriver.Chrome(
        executable_path=webdriver_path)  # 注意改你安装插件的路径
    browser.get(url)
    time.sleep(2)
    browser.find_element_by_xpath('//*[@id="j_username"]').send_keys('tiger1575344246')
    browser.find_element_by_xpath('//*[@id="j_password_show"]').send_keys('Tiger06191027')
    # --------------获取登陆验证码------------------
    imgElement = browser.find_element_by_id('codePic')  # 通过截图获取验证码图片
    get_auth_code(browser, imgElement)
    # code = input('请输入验证码：')  # 手动
    code = validation_result()  # 打码平台/调试要花钱，尽量手动输入验证码

    # now_handle = browser.current_window_handle
    browser.find_element_by_xpath('//*[@id="j_validation_code"]').send_keys(str(code))
    browser.find_element_by_xpath('//*[@id="globleBody"]/div[2]/div/div[1]/div[3]/div[2]/div[2]/div/a[1]').click()
    # -------waiting for success login--------
    # 该元素为 登录状态 判断标志
    try:
        WebDriverWait(browser, 10, ignored_exceptions=None).until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="globleBody"]/div[2]/div/div[1]/div[3]/div[2]/div[1]/div[3]/div/div[1]/a/img'))
        )
        print("登陆成功!")
        time.sleep(2)
        # print( browser.current_url)
        # cc = browser.find_element_by_xpath('//*[@id="quickInput"]')
        # print(cc)
        browser.find_element_by_xpath('//*[@id="quickInput"]').send_keys(key)
        browser.find_element_by_xpath('//*[@id="quickSearch"]').click()
    except selenium.common.exceptions.TimeoutException:
        print('TimeoutException')
        return False

    for handle in browser.window_handles:  # 始终获得当前最后的窗口
        browser.switch_to.window(handle)
        time.sleep(2)
    # -------waiting for success loading result--------
    # 该元素为 搜索结果 加载完毕判断标志
    try:
        element = WebDriverWait(browser, 20, ignored_exceptions=None).until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="resultMode"]/div/div[1]/ul/li[1]/div/div[3]/div/a[1]'))
        )
        print('加载完成')
        js = "var q=document.documentElement.scrollTop=800"
        browser.execute_script(js)
        time.sleep(2)
        element.click()
    except selenium.common.exceptions.TimeoutException:
        print('TimeoutException')
        return False
    # browser.find_element_by_xpath('//*[@id="resultMode"]/div/div[1]/ul/li[1]/div/div[3]/div/a[1]').click()

    for handle in browser.window_handles:  # 始终获得当前最后的窗口
        time.sleep(1)
        browser.switch_to.window(handle)
    # -------waiting for success loading details--------
    # 该元素为 详情页面 加载完毕判断标志
    try:
        element = WebDriverWait(browser, 10, ignored_exceptions=None).until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="patent_list"]/li/div[3]/a[1]'))
        )
        element.click()
    except selenium.common.exceptions.TimeoutException:
        print('TimeoutException')
        return False
    # ------------------获取下载验证码--------------
    time.sleep(1)
    get_download_code(browser)
    # download_code = input('请输入下载验证码：')
    download_code = str(validation_result())  # 调用打码平台的结果
    # 这里的真正输入框是第二个元素
    b = browser.find_elements_by_xpath('//*[@id="downloadValidateCode"]')
    time.sleep(1)
    b[1].send_keys(str(download_code))
    # ---------点击下载按钮----------
    browser.find_element_by_xpath('/html/body/div[7]/div/table/tbody/tr[3]/td/div[2]/button[2]').click()
    time.sleep(5)
    return True


'''解压.zip，放到filedir'''
def release(fromfile, tofile):
    r = zipfile.is_zipfile(fromfile)
    if r:
        starttime = time.time()
        fz = zipfile.ZipFile(fromfile, 'r')
        for file in fz.namelist():
            # print(file)  #打印zip归档中目录
            fz.extract(file, tofile)
        endtime = time.time()
        # times = endtime - starttime
    else:
        print('This file is not zip file')
    print('this file has been released')


def save(txt_path, contents):
    fh = open(txt_path, 'w', encoding='utf-8')
    fh.write(contents)
    fh.close()


'''把html转txt，获取结构化信息'''
def get_info(filedir):
    ablist = []
    applydate = []
    gongkaidate = []
    ipclist = []
    gongkainum = []
    people = []
    applypeople = []
    message_list = []
    # release(r'C:\Users\whh\Downloads\\downfile.zip',filedir)
    # get_txt(filedir)
    file = os.listdir(filedir)[0]
    # #print(file)
    path = 'd:\\testData\\' + file
    sub_file = os.listdir(path)[0]
    # print(sub_file)
    htm = path + '\\' + sub_file
    htmlf = open(htm, 'r', encoding="utf-8")
    htmlcont = htmlf.read()
    # print(htmlcont)
    # 摘要
    re_abstract = r'base=.*<'
    abstract = re.findall(re_abstract, htmlcont)
    re_more0 = r'[\u4e00-\u9fa5，、；]+'
    abstract0 = re.findall(re_more0, abstract[0])
    for i in abstract0:
        ablist.append(i)
    # print(ablist)

    # 申请日,公告日
    re_apply = r'[0-9]+\.[0-9]+\.[0-9]+'
    date = re.findall(re_apply, htmlcont)
    applydate.append(date[0])
    gongkaidate.append(date[1])

    # ipc分类号
    re_ipc = r'IPC分类号<td>\n.*'
    ipc = re.findall(re_ipc, htmlcont)
    re_more1 = r'[0-9A-Z//]+'
    num = re.findall(re_more1, ipc[0])
    ipclist.append(num[4])
    # print(ipclist)

    # 公开号
    re_ipc = r'公开（公告）号<td>\n.*'
    number = re.findall(re_ipc, htmlcont)
    re_more1 = r'[0-9A-Z//]+'
    a = re.findall(re_more1, number[0])
    # print(num)
    gongkainum.append(a[3])

    # 发明人
    re_ipc = r'发明人<td>\n.*'
    b = re.findall(re_ipc, htmlcont)
    re_more1 = r'[\u4e00-\u9fa5，、；]+'
    person = re.findall(re_more1, b[0])
    # print(num)
    people.append(person[1])

    # 申请人
    re_ipc = r'申请（专利权）人<td>\n.*'
    b = re.findall(re_ipc, htmlcont)
    re_more1 = r'[\u4e00-\u9fa5，、；]+'
    person1 = re.findall(re_more1, b[0])
    # print(person1)
    applypeople.append(person1[3])

    length = len(gongkainum)
    for i in range(0, length):
        message = [gongkainum[i], applydate[i], gongkaidate[i], ipclist[i], people[i], applypeople[i], ablist[i]]
        message_list.append(message)
        print(message_list)
    return message_list


'''把html文件转txt，获取全文的文本'''
def get_txt(dadpath):
    file = os.listdir(dadpath)[0]
    # print(file)
    path = 'd:\\testData\\' + file
    sub_file = os.listdir(path)[2]
    # print(sub_file)
    htm = path + '\\' + sub_file
    htmlf = open(htm, 'r', encoding="utf-8")
    htmlcont = htmlf.read()
    # print(htmlcont.text())
    html_text = re.sub("[A-Za-z0-9\!\%\[\]\,\。\< \> \/ \"=\" \_ \= \: \{.*} \; \---'宋体'--#---'宋体'--#?-?]", "", htmlcont)
    # print(html_text)
    save_path = 'd:\\testData\\' + file + '\\000.txt'
    save(save_path, html_text)
    with open(save_path, 'r', encoding='UTF-8') as f:
        return str(f.read())


if __name__ == '__main__':
    data = from_excel(file_path)
    count = 0
    # key = sys.argv[1] # 与网页中php文件调用结合
    for i in range(5500, 5505):  # 表格中下载的目标文件范围
        number = data[i][0]
        key = change_str(str(number))
        if get_download(targeturl, key):
            count += 1
            print("All downloads is", count)
        else:
            continue
    # release(filename, filedir)
    # print(get_txt(filedir))