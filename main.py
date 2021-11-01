# _*_coding:utf-8_*_

from PyQt5.QtWidgets import *

from PyQt5.QtCore import pyqtSignal, QRect
import sys

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx
import matplotlib.pyplot as plt


class Figure_Canvas(FigureCanvas):
# 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，
# 这是连接pyqt5与matplotlib的关键
    select_Value = pyqtSignal()

    def __init__(self):
        self.fig = plt.figure()  # 可选参数,facecolor为背景颜色
        FigureCanvas.__init__(self, self.fig)  # 初始化激活widget中的plt部分
        self.pos = []  # 保存每个节点的地址
        self.model_struct = []  # 所有节点信息
        self.focus_loc = []  # 现在的焦点
        self.train_loc = []  # 训练文件路径
        self.struct_loc = []  # 结构文件路径

    def draw_graph(self):
        with open(self.struct_loc, "r") as f:  # 打开文件
            data = f.read()  # 读取文件
        self.model_struct = BayesianModel(eval(data))  # 8-9
        #如果不使用canvas.mpl_connect的话将不能激活event
        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_press(event))

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        self.pos = nx.planar_layout(self.model_struct)
        nx.draw(self.model_struct, self.pos, with_labels=True, node_color='blue', edge_color='green', node_size=1500)
        # nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)

    def on_press(self, event):  # 更新表格的CPD
        loc_dis = list([])
        min_dis = 100  # 将最小距离设定为100
        possible_loc = []  # 将最可能位置设定为空
        for element in self.pos:
            temp_dis = (event.xdata-self.pos[element][0])**2+(event.ydata-self.pos[element][1])**2
            loc_dis.append({element: temp_dis})
            if min_dis > temp_dis:
                min_dis = temp_dis
                possible_loc = element
            else:
                continue
        self.focus_loc = possible_loc
        self.select_Value.emit()

    def train_cpds(self):
        data = pd.read_csv(self.train_loc, sep='\t')
        samples = data[:6301]
        samples['ex'] = samples['ex'].map(lambda x: x + 1)
        samples['su'] = samples['su'].map(lambda x: x + 1)
        mle = MaximumLikelihoodEstimator(model=self.model_struct, data=samples)
        parameter_table = mle.get_parameters()
        for element in parameter_table:
            self.model_struct.add_cpds(element)


class Example(QMainWindow):

    def __init__(self):
        super(Example, self).__init__()
        self.dr = Figure_Canvas()
        self._initUI()

    def _initUI(self):

        self.dr.select_Value.connect(self.refresh_table)

        # 实例化主窗口的QMenuBar对象
        bar = self.menuBar()
        # 向菜单栏中添加新的QMenu对象，父菜单
        file = bar.addMenu('开始')
        # 向QMenu小控件中添加按钮，子菜单
        train_file = file.addAction('添加训练文件')
        # 创建新的子菜单项，并添加孙菜单
        struct_file = file.addAction('添加结构文件')

        center_widget = QWidget()  # 设置中心控件

        bt1 = QPushButton('开始训练')
        bt2 = QPushButton('开始绘图')
        bt1.clicked.connect(self.train_model)
        bt2.clicked.connect(self.draw_network)
        self.network_window = QGraphicsView()
        self.node_table = QTableWidget()
        self.struct_label = QLabel()
        self.train_label = QLabel()
        # 单击任何Qmenu对象，都会发射信号，绑定槽函数
        train_file.triggered.connect(self.get_train_file)
        struct_file.triggered.connect(self.get_struct_file)
        # 建立顶层控件
        wlayout = QVBoxLayout()
        # 局部布局：水平，垂直，网格，表单
        hlayout = QHBoxLayout()
        vlayout = QVBoxLayout()
        label_layout = QFormLayout()
        # 为局部布局添加控件
        hlayout.addWidget(bt1)
        hlayout.addWidget(bt2)

        vlayout.addWidget(self.network_window)
        vlayout.addWidget(self.node_table)

        label_layout.addRow('结构文件路径', self.struct_label)
        label_layout.addRow('训练文件路径', self.train_label)
        # 准备四个控件
        hwg = QWidget()
        vwg = QWidget()
        fwg = QWidget()

        # 使用四个控件设置局部布局
        hwg.setLayout(hlayout)
        vwg.setLayout(vlayout)
        fwg.setLayout(label_layout)

        # 将四个控件添加到全局布局中
        wlayout.addWidget(vwg)
        wlayout.addWidget(hwg)
        wlayout.addWidget(fwg)

        # 将窗口本身设置为全局布局

        center_widget.setLayout(wlayout)
        self.setCentralWidget(center_widget)

        self.show()

    def draw_network(self):
        self.dr.draw_graph()  # 画图, 并返回每个点的位置
        # 第四步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        graphicscene = QGraphicsScene(0, 0, 800, 800)
        # 第五步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        graphicscene.addWidget(self.dr)
        # 第六步，把QGraphicsScene放入QGraphicsView
        self.network_window.setScene(graphicscene)
        # 第七步，调用show方法呈现图形！
        self.network_window.show()

    def get_train_file(self):
        text = QFileDialog.getOpenFileName(None, "训练文件", ".", "*.tsv")
        self.train_label.resize(200, 20)
        self.train_label.setText(text[0])
        self.dr.train_loc = text[0]

    def get_struct_file(self):
        text = QFileDialog.getOpenFileName(None, "结构文件", ".", "*.txt")
        self.struct_label.resize(200, 20)
        self.struct_label.setText(text[0])
        self.dr.struct_loc = text[0]

    def train_model(self):  # 训练数据
        self.dr.train_cpds()

    def processtrigger(self):
        print("训练完毕")

    def refresh_table(self):
        for element in self.dr.model_struct.cpds:
            if element.variable == self.dr.focus_loc:
                if len(element.values.shape) < 3:  # 只显示2维以及以下的条件概率表
                    value_str = element.variables[1:len(element.variables)]  # 变量名
                    main_str = [element.variable]  # 主变量名
                    # 变量状态数
                    col_name = [x+str(y) for x in value_str for y in element.state_names[value_str[0]]]
                    row_name = [x+str(y) for x in main_str for y in element.state_names[main_str[0]]]
                    self.node_table.setRowCount(len(row_name))
                    self.node_table.setColumnCount(len(col_name))
                    self.node_table.setHorizontalHeaderLabels(col_name)  # 设置列名称
                    self.node_table.setVerticalHeaderLabels(row_name)  # 设置行名称
                    for main_status in range(element.values.shape[0]):
                        for other_status in range(element.values.shape[1]):
                            newItem = QTableWidgetItem(str(element.values[main_status, other_status]))
                            self.node_table.setItem(int(main_status), int(other_status), newItem)
                break
            else:
                continue


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
