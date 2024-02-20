#!/usr/bin/python3
# -*- coding: utf-8 -*-


import csv
import numpy as np

filex = open('x.csv', 'w', encoding='gbk', newline='')
csv_writer = csv.writer(filex)
# 3. 构建列表头
csv_writer.writerow(["大小", "人口", "新旧程度", "天然气普及程度", "电价", "人均收入", "人均居住面积", "年平均温度", "年最高温度", "年最低温度"])
# 4. 写入csv文件内容
for i in range(1000):
    csv_writer.writerow([
        np.random.randint(100),
        np.random.randint(100),
        np.random.randint(100),
        np.random.randint(100),
        np.random.randint(100),
        np.random.randint(100),
        np.random.randint(100),
        np.random.randint(100),
        np.random.randint(50, 100),
        np.random.randint(50)
    ])
filex.close()

filey = open('y.csv', 'w', encoding='gbk', newline='')
csv_writer_y = csv.writer(filey)
csv_writer_y.writerow(
    ["1月用电量", "2月用电量", "3月用电量", "4月用电量", "5月用电量", "6月用电量", "7月用电量", "8月用电量", "9月用电量", "10月用电量", "11月用电量", "12月用电量", ])
for i in range(1000):
    csv_writer_y.writerow(np.random.rand(12) * 100)
filey.close()
