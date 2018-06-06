# End_to_End_Learning_for_Self_Driving_Cars
tensorflow版本 1.8 <br>
用tensorflow实现[论文](https://arxiv.org/pdf/1604.07316.pdf) <br>
代码参考了[SullyChen/Autopilot-TensorFlow](https://github.com/SullyChen/Autopilot-TensorFlow) 感谢！<br>
训练数据[dataset.zip](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view) <br>
将model.ipynb打开运行到底即可，optimize函数决定了训练次数。<br>
使用tensorboard: 打开terminal，输入 tensorboard --logdir './graph' <br>
要可视化卷积层和卷积核，请使用代码中定义的两个helper function<br>
