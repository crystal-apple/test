import requests

url = 'http://2.2.2.2/ac_portal/login.php'
body = {"opr": "pwdLogin", "userName": "张三", "pwd": "密码", "rememberPwd": "0"}

response = requests.post(url, data=body)
print(response.text)
