#QThread就是PyQt5提供的线程类
#由于是一个完成了的类，功能已经写好了，线程类线程的功能需要我们自己完成
#需要自己完成需要的线程类，创建一个新的线程类（功能就可以自己定义），继承QThread
#新写的类就是线程类
import base64
import sqlite3
import cv2
import requests
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, QDateTime

#线程类进行执行只会执行线程类中的run函数，如果有新的功能需要实现的话，重新写一个run函数完成
from PyQt5.QtWidgets import QInputDialog



class detect_thread(QThread):
    transmit_data = pyqtSignal(dict)#自定义信号槽
    search_data = pyqtSignal(str)
    OK = True
    #字典用来存放签到数据
    sign_list = {}

    def __init__(self,token,group_id):
        super(detect_thread,self).__init__()#初始化操作
        self.access_token = token
        self.group_id = group_id
        self.condition = False
        self.add_status = 0
        # self.create_sqlite()



    #run函数执行结束，代表线程结束
    def run(self):
        print("run")
        '''
        self.time = QTimer(self)
        self.time.start(500)
        self.time.timeout.connect(self.detect_face)
        '''
        while self.OK:
            if self.condition:
                self.detect_face(self.base64_image)
                self.condition = False
        print("while finish")

    def get_base64(self,base64_image):
        #当窗口产生信号，调用槽函数，就把传递的数据，存放在线程的变量中
        self.base64_image = base64_image
        self.condition = True

    # def get_group_id(self,group_id):atetime):
        con = sqlite3.connect(r"stu_data.db")
        c = con.cursor()
    #     self.group_id = group_id



    def detect_face(self,base64_image):
        '''
        #对话框获取图片
        #获取一张图片（一帧画面）
        #getOpenFileName通过对话框的形式获取一个图片（.JPG）路径
        path,ret = QFileDialog.getOpenFileName(self,"open picture",".","图片格式(*.jpg)")
        #把图片转换成base64编码格式
        fp = open(path,'rb')
        base64_imag = base64.b64encode(fp.read())
        print(base64_imag)
        '''
        # 摄像头获取画面
        # camera_data = self.cameravideo.read_camera()
        # # 把摄像头画面转换成图片，然后设置编码base64编码格式数据
        # _, enc = cv2.imencode('.jpg', camera_data)
        # base64_image = base64.b64encode(enc.tobytes())
        # 发送请求的地址
        request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
        # 请求参数是一个字典，在字典中存储,百度AI要识别的图片信息，属性内容
        params = {
            "image": base64_image,  # 图片信息字符串
            "image_type": "BASE64",  # 图片信息格式
            "face_field": "gender,age,beauty,expression,face_shape,glasses,emotion,mask",  # 请求识别人脸的属性， 各个属性在字符串中用,逗号隔开
            "max_face_num": 1
        }
        # 访问令牌
        access_token = self.access_token
        # 把请求地址和访问令牌组成可用的网络请求
        request_url = request_url + "?access_token=" + access_token
        # 参数：设置请求的格式体
        headers = {'content-type': 'application/json'}
        # 发送网络post请求,请求百度AI进行人脸检测,返回检测结果
        # 发送网络请求，就会存在一定的等待时间，程序就在这里阻塞执行，所以会存在卡顿现象
        response = requests.post(request_url, data=params, headers=headers)
        if response:
            data = response.json()
            if data['error_code'] != 0:
                self.transmit_data.emit(data)
                self.search_data.emit(data['error_msg'])
                return

            if data['result']['face_num'] > 0:
                #data是请求数据的结果，需要进行解析，单独拿出所需的数据内容，分开
                self.transmit_data.emit(dict(data))
                self.face_search(self.group_id)


    # 人脸识别检测，只检测一个人
    def face_search(self,group_id):
        request_url = "https://aip.baidubce.com/rest/2.0/face/v3/search"
        params = {
            "image": self.base64_image,
            "image_type": "BASE64",
            "group_id_list": group_id,
        }
        access_token = self.access_token
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/json'}
        response = requests.post(request_url, data=params, headers=headers)
        if response:
            data = response.json()
            if data['error_code'] == 0:
                if data['result']['user_list'][0]['score'] > 90:
                    #存储要保存的签到数据，方便显示
                    del(data['result']['user_list'][0]['score'])
                    datetime = QDateTime.currentDateTime()
                    datetime = datetime.toString()
                    data['result']['user_list'][0]['datetime'] = datetime
                    key = data['result']['user_list'][0]['group_id']+data['result']['user_list'][0]['user_id']
                    if key not in self.sign_list.keys():
                        self.sign_list[key] = data['result']['user_list'][0]
                    self.search_data.emit("学生签到成功\n学生信息："+data['result']['user_list'][0]['user_info'])
                    stu_data = data['result']['user_list'][0]['user_info']
                    info = stu_data.split('\n')
                    _, info_name = info[0].split('：')
                    _, info_class = info[1].split('：')
                    id = data['result']['user_list'][0]['user_id']
                    # self.add_sqlite(id, info_name, info_class, datetime)
      #              self.search_sqlite(id)
        #            for i in self.values:
        #                search_id = i[0]
        #            if search_id == id:
         #               self.update_sqlite(id,info_name,info_class,datetime)
       #             else:
       #                 self.add_sqlite(id,info_name,info_class,datetime)
                else:
                    self.search_data.emit("学生签到失败，找不到对应学生")

    #创建数据库
    def create_sqlite(self):
        con = sqlite3.connect(r"stu_data.db")
        c = con.cursor()
        c.execute("create table student(id primary key ,name ,stu_class,datetime)")
        print("创建成功")

    #添加学生数据到数据库
    def add_sqlite(self,id,name,stu_class,datetime):
        con = sqlite3.connect(r"stu_data.db")
        c = con.cursor()
        value = (id,name,stu_class,datetime)
        sql = "insert into student(id,name,stu_class,datetime) values(?,?,?,?)"
        c.execute(sql,value)
        print("添加成功")
        # 提交
        con.commit()

    #更新学生数据库信息
    def update_sqlite(self,id,name,stu_class,datetime):
        con = sqlite3.connect(r"stu_data.db")
        c = con.cursor()
        # value = (name,stu_class,datetime,id)
        sql = "update student set name=?,stu_class=?,datetime=? where id =?"
        c.execute(sql,(name,stu_class,datetime,id))
        con.commit()
        print("更新成功")

    #查询学生数据库信息
    def search_sqlite(self,id):
        con = sqlite3.connect(r"stu_data.db")
        c = con.cursor()
        sql = "select * from student where id=?"
        self.values = c.execute(sql,(id,))
