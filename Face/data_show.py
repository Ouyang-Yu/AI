from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QAbstractItemView, QFileDialog
from sign_indata import Ui_Dialog


class sign_data(Ui_Dialog,QDialog):
    def __init__(self, signdata,parent=None):
        super(sign_data, self).__init__(parent)
        self.setupUi(self)#创建界面内容
        #设置窗口内容不能被修改
        self.signdata = signdata
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i in signdata.values():
            info = i['user_info'].split('\n')
            rowcount = self.tableWidget.rowCount()
            self.tableWidget.insertRow(rowcount)
            info_name = info[0].split('：')
            self.tableWidget.setItem(rowcount, 0, QTableWidgetItem(info_name[1]))
            info_class = info[1].split('：')
            self.tableWidget.setItem(rowcount, 1, QTableWidgetItem(info_class[1]))
            self.tableWidget.setItem(rowcount, 2, QTableWidgetItem(i['user_id']))
            self.tableWidget.setItem(rowcount, 3, QTableWidgetItem(i['datetime']))
        #导出按钮
        self.pushButton.clicked.connect(self.save_data)
        #取消按钮
        self.pushButton_2.clicked.connect(self.close_window)
    def close_window(self):
        self.reject()

    def save_data(self):
        #打开对话框，获取要导出的数据文件名
        filename,ret = QFileDialog.getSaveFileName(self,"导出数据",".","TXT(*.txt)")
        f = open(filename,"w")
        for i in self.signdata.values():
            info = i['user_info'].split('\n')
            _,info_name = info[0].split('：')
            _,info_class = info[1].split('：')
            f.write(str(info_name+"  "+info_class+"  "+i['user_id']+"  "+i['datetime'] ))
        f.close()
        self.accept()