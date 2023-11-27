# -*- coding: utf-8 -*-
# @Author: Yzm
# @Date:   2021-09-24 22:16:44
# @Last Modified by:   Yzm
# @Last Modified time: 2021-11-21 16:32:22
from astropy.io import fits
import os
import numpy as np
import math
import pandas as pd
import sys
# import operator
import matplotlib.pyplot as plt
# from scipy import arange
from PIL import Image, ImageDraw, ImageFont
import re


class SExtactor(object):
    """
    Param docstring for SExtactor
    fitPath:输入图片路径
    sexPath：image.sex路径（image.sex为SExtractor自带的存储扫描结果的文件）
    configFile：SExtractor配置文件（daofind.sex）
    int_switch：是否将扫描结果作为整数，True：整数，False：小数
    password：服务器密码
    catalog：扫描结果转存的list
    row_info：扫描用到的参数（default.param中的参数）
    """

    def __init__(self, fitPath, sexPath, configFile, paramFile, int_switch, password):
        super(SExtactor, self).__init__()
        self.fitPath = fitPath
        self.sexPath = sexPath
        self.configFile = configFile
        self.paramFile = paramFile
        self.int_switch = int_switch
        self.password = password

    # self.catalog, self.row_info = self.read_catalog()

    def add_attributes(self):
        self.fits_array = self.read_fits()
        self.dataframe = self.catalog_dataframe()
        self.numbers = len(self.catalog)

    @property
    def read_configue(self):
        '''[读SExtractor配置文件参数信息并转为Dataframe]

        [description]

        Returns:
            [Dataframe] -- [Dataframe的shape：（配置文件行数，3）
                            第一列'line'：对应.sex文件的行号，
                            第二列'param'：为参数keyworld
                            第三列'value'：为参数取值]
        '''
        conf_list = []
        with open(self.configFile, 'r') as f:
            context = f.readlines()
        for index, line in enumerate(context):
            if len(line) < 10 or line[0] == '#' or line.split()[0] == '#':
                pass
            else:
                info = line.split('#')[0].split()
                conf_dict = {}
                # print(line.split('#')[0].split())
                conf_dict['line'] = index
                conf_dict['param'] = info[0]
                conf_dict['value'] = ''.join(info[1:])
                conf_list.append(conf_dict)
        # pprint(conf_list)
        return pd.DataFrame(conf_list, columns=['line', 'param', 'value'])

    @property
    def catalog_dataframe(self):
        '''创建Dataframe'''
        return pd.DataFrame(self.catalog, columns=self.row_info)

    @property
    def read_param(self):
        param_list = []
        with open(self.paramFile, 'r') as f:
            text = f.readlines()
        for index, line in enumerate(text):
            if len(line) > 2:
                info = line.split()[0].split('#')
                # print(info)
                param_dict = {}
                param_dict['line'] = index
                param_dict['param'] = info[-1]
                param_dict['value'] = True if len(info[0]) > 1 else False
                # print(param_dict)
                param_list.append(param_dict)
        # pprint(conf_list)
        return pd.DataFrame(param_list, columns=['line', 'param', 'value'])

    @property
    def show_on_param(self):
        df = self.read_param
        # return df[df.value == True].values.tolist()
        # param_dtype = np.dtype([('line', 'i2'), ('key', 'S20'), ('value', 'b')])
        # return np.array(df[df.value == True])
        return df[df.value == True]

    def cat_keyworld(self, keyworld, flag):
        """[查看指定参数的信息]

        [description]

        Arguments:
            keyworld {[str]} -- [参数关键字]
            flag {[str]} -- [配置或参数标志]

        Returns:
            [list] -- [指定参数的信息['line', 'param', 'value']]
        """
        if flag == 'sex':
            df = self.read_configue
            return df[df.param == keyworld].values.tolist()
        elif flag == 'param':
            df = self.read_param
            return df[df.param == keyworld].values.tolist()
        else:
            print('flag input error！！')

    def set_configue(self, keys, values):
        """[设置参数]

        [description]

        Arguments:
            keys {[list]} -- [需修改的参数关键字列表]
            values {[list]} -- [对应的修改的参数取值]
        """
        with open(self.configFile, 'r') as f:
            context = f.readlines()
        for key, value in zip(keys, values):
            line = self.cat_keyworld(key, 'sex')[0][0]
            # print('修改参数所在行', line)
            # print('切的结果', context[line].split('#', 1))
            info, annotation = context[line].split('#', 1)
            column1 = format(key, '<17')
            column2 = format(str(value), '<15')
            column3 = '#' + annotation
            context[line] = "{}{}{}".format(column1, column2, column3)
        # print('新行', context[line])
        # pprint(context)
        with open(self.configFile, 'w') as f:
            f.write(''.join(context))

    def set_param(self, key, state):
        """[设置参数]

        [description]

        Arguments:
            key {[str]} -- [需修改的参数关键字]
            state {[bool]} -- [参数状态]
        """
        with open(self.paramFile, 'r') as f:
            context = f.readlines()
        # if key in context.split():
        print(self.read_param.param)
        if key in np.array(self.read_param.param):
            line, _, value = self.cat_keyworld(key, 'param')[0]
            # print(type(state), type(value))
            if value == state:
                print('***param do not need modify!!***')
                return None
            context[line] = key + '\n' if state is True else '#{}\n'.format(key)
            with open(self.paramFile, 'w') as f:
                f.write(''.join(context))
        else:
            if state is False:
                print('***param do not exist!!!, do not need set to False***', '\n你在淦神魔？')
                return None
            with open(self.paramFile, 'a') as f:
                f.write(key)

    def call_sex(self):
        """调用SExtractor"""
        # os.system('echo %s | sudo -S %s' % (self.password, 'sextractor -c ' + self.configFile + self.fitPath))
        os.system('echo %s | sudo -S sextractor -c %s %s' % (self.password, self.configFile, self.fitPath))

    def read_catalog(self):
        """提取image.sex文件的坐标和列信息"""
        self.call_sex()
        # sextr(self.fitPath, self.configFile, 741236985)
        coords = []
        with open(self.sexPath, 'r') as sex_file:
            old = sex_file.readlines()
        raw_info = [line.split()[2] for line in old if line[0] == '#']
        # print('让我看看raw_info', raw_info)
        old = old[len(raw_info):]
        # print('坐标第一行', old[0])

        if self.int_switch is True:
            # coords = [list(map(int, map(float, line.split()))) for line in old if line[0] != '#']
            coords = [tuple(map(int, map(float, line.split()))) for line in old if line[0] != '#']
        elif self.int_switch is False:
            # coords = [list(map(float, line.split())) for line in old if line[0] != '#']
            coords = [tuple(map(float, line.split())) for line in old if line[0] != '#']

        return coords, raw_info

    def judge_param(self, param_list):
        """判断SExtractor配置文件采参数是否齐全"""
        judge_condition = [x for x in param_list if x not in self.row_info]
        if judge_condition == []:
            pass
        else:
            print('请在SExtractor配置文件中添加%s参数' % judge_condition)
            sys.exit()

    def read_fits(self):
        with fits.open(self.fitPath) as hdu:
            image = hdu[0].data
        return image

    def creat_stamp(self, image, X, Y, RADIUS):
        """给定中心半径截stamp"""
        return image[(round(Y) - math.ceil(RADIUS)):(round(Y) + math.ceil(RADIUS) + 1),
               (round(X) - math.ceil(RADIUS)):(round(X) + math.ceil(RADIUS) + 1)]


class ColdHotdetector(SExtactor):
    """docstring for ColdHotdetector"""

    def __init__(self, fitPath, sexPath, configFile, paramFile, int_switch, password, cold_keys, cold_values):
        super(ColdHotdetector, self).__init__(fitPath, sexPath, configFile, paramFile, int_switch, password)
        self.cold_keys = cold_keys
        self.cold_values = cold_values

    @property
    def cold(self):
        """
        [大阈值检测]
        Returns:
            [list] -- [description]
        """
        self.set_configue(self.cold_keys, self.cold_values)
        catalog = self.read_catalog()[0]
        return catalog

    def array_type(self):
        # 结构化矩阵自定义dtype
        params = self.show_on_param.param.tolist()
        type_list = []
        for index, param in enumerate(params):
            if index <= 1:
                type_list.append((param, 'int'))
            else:
                type_list.append((param, 'f4'))

        return np.dtype(type_list)

    def draw(self, jpg, R, save):
        # 读fits
        array = self.read_fits()
        catalog_dtype = self.array_type()
        print(catalog_dtype)
        cold = np.asarray(self.cold, catalog_dtype)
        # self.bbox_by_kron(array, cold, 'red', R, save)
        self.draw_bbox(jpg, cold, save, R, R, 'red', 2)

    def draw_bbox(self, jpg, starList, save, x, y, color, lineWidth):
        """根据坐标绘制bbox，由于fits图像与jpg图像原点不同，需要两次翻转"""
        Image.MAX_IMAGE_PIXELS = 1000000000
        with Image.open(jpg) as image:
            out = image.transpose(Image.FLIP_TOP_BOTTOM)
            draw = ImageDraw.Draw(out)
            for i in starList:
                bbox = [i[0] - x, i[1] - y, i[0] + x, i[1] + y]
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=lineWidth)
            del draw
            out2 = out.transpose(Image.FLIP_TOP_BOTTOM)
        out2.save(save, overwrite=True)
        print('draw complete')

    def bbox_by_kron(self, image, starList, color, Width, save):
        fig = plt.figure(figsize=(22, 22))  # 单位为100
        # ax = fig.add_subplot(1,1,1)
        ax = plt.gca()

        for index, each in enumerate(starList):
            top_left_x, top_left_y = each['X_IMAGE'] - each['FLUX_RADIUS'] - 4, each['Y_IMAGE'] - each[
                'FLUX_RADIUS'] - 4
            rect = plt.Rectangle((top_left_x, top_left_y), each['FLUX_RADIUS'] * 2 + 6, each['FLUX_RADIUS'] * 2 + 6,
                                 fill=False, edgecolor=color, linewidth=Width)
            ax.add_patch(rect)
        # context = (round(each['CXX_IMAGE'], 2), round(each['CYY_IMAGE'], 2), round(each['CXY_IMAGE'], 2))
        # context = (round(each['CXX_IMAGE']/each['CYY_IMAGE'], 2), round(each['CXY_IMAGE'], 2))
        # context = (round(each['CXX_IMAGE']/each['CYY_IMAGE'], 2))
        # ax.text(top_left_x, top_left_y, context, fontsize=16, color="r", style="italic", weight="light", verticalalignment='center', horizontalalignment='right', rotation=0)

        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(save)
    # plt.show()


if __name__ == '__main__':
    #TODO: OK, the following part needs to be redefined to relative positions or config files. PENG 20231101
    for filename in os.listdir(r"/home/lab30202/sdc/YangWang_1/UV/datas/Ori_Datas/fits"):  # listdir的参数是文件夹的路径
        # img = fits.open("zw1/" + filename)
        fitPath = "/home/lab30202/sdc/YangWang_1/UV/datas/Ori_Datas/fits/" + filename
        # jpg = '/home/deng/yangwang/chuli/uV_2021-06-15T07%3A17%3A00.293_4.000s/uV_2021-06-15T07%3A17%3A00.293_4.000s.png'
        name = os.path.basename(fitPath).split('.fit')[0]
        keys = ['DETECT_TYPE',
                'DETECT_MINAREA',
                'DETECT_THRESH',
                'ANALYSIS_THRESH',
                'DEBLEND_NTHRESH',
                'DEBLEND_MINCONT',
                'CLEAN_PARAM',
                'BACK_SIZE',
                'BACK_FILTERSIZE']

        values = ['CCD', 3, 1.5, 1.5, 32, 0.005, 1.0, 128, 3]
        sex = ColdHotdetector(
            fitPath, 'image2.sex', 'simplify.sex', 'test.param', True, '741236985',
            keys, values)
        result=sex.cold
        # print(type(result))
        result=[i[:2] for i in result]
        result = np.array(result, dtype='int_')
        print(result)
        # np.savetxt(name+".npy", result,fmt='%d')
        image=sex.read_fits()
        # # print(image.shape)
        mask=np.ones((2048,2048))
        for i in range(result.shape[1]):
            a=result[i]
            mask[a[0]-5:a[0]+5,a[1]+5:a[1]-5]=0 #截取一块矩阵
        x,y=[],[]
        with open ('outlier.data') as A:
            for eachline in A:
                tmp=re.split("\s+",eachline.rstrip())
                x.append(tmp[0])
                y.append(tmp[1])
        for i in range(len(x)):
            mask[int(x[i]),int(y[i])]=0
        MSE = (np.square(np.multiply(mask, image))).mean()
        print(MSE)
        num=0
        time=4
        with fits.open('2.fits') as hdu:
            img1 = hdu[0].data
        MSE1 = (np.square(np.multiply(mask, img1))).mean()
        select1=abs(MSE1-MSE)
        #TODO: OK, the following part needs to be redefined to relative positions or config files. PENG 20231101
        for filename1 in os.listdir(r"/home/lab30202/sdc/lvchao/zw1"):  # listdir的参数是文件夹的路径
            img = fits.open("/home/lab30202/sdc/lvchao/zw1/" + filename1)
            img2 = img[0].data
            img2=np.array(img2)
            num=num+1
            MSE2 = (np.square(np.multiply(mask, img1))).mean()
            select2=abs(MSE2-MSE)
            if (select1 > select2):
                select1 = select2
                time=num*4
        mask1=np.loadtxt('20211116150236mask2.npy')
        func=np.loadtxt('20211116150236f1_arr.npy')
        for i in range(func.shape[1]):
            b=func[i]
            x=time
            y = b[0] * pow(x, 10) + b[1] * pow(x, 9) + b[2] * pow(x, 8) + b[3] * pow(x, 7) + b[4] * pow(x, 6) + b[
                5] * pow(x, 5) + b[6] * pow(x, 4) + b[7] * pow(x, 3) + b[8] * pow(x, 2) + b[9] * pow(x, 1) + b[10]
            num = i  # 想要替换的数字
            NUM = y  # 替换后的数字
            index = (mask1 == num)
            mask1[index] = NUM
        mask1[np.isnan(mask1)]=0
        image=image-mask1
        hdu = fits.PrimaryHDU(image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(filename)
        # with fits.open('29.fits') as hdu:
        #     image1 = hdu[0].data
        # MSE1=(np.square(np.multiply(mask, image1))).mean()
        # print(MSE1)


    # R = 5
    # save_path='/home/deng/yangwang/chuli'
    # save = os.path.join(save_path, name+'.jpg')
    # save = '/home/deng/yangwang/chuli/uV_2021-06-15T07%3A17%3A00.293_4.000s/uV_2021-06-15T07%3A17%3A00.293_4.000s.jpg'
    # sex.draw(jpg, R, save)
