#include <opencv2/opencv.hpp>   //OpenCV头文件
#include <iostream>
using namespace cv;   //启用 OpenCV名称空间
using namespace std;

bool selectObject = false;  //用于标记是否有选取目标
int trackObject = 0; //1 表示有追踪对象 0 表示无追踪对象 -1 表示追踪对象尚未计算 Camshift 所需的属性
Rect selection;   //保存鼠标选择区域
Mat image;     //缓存读取到的视频帧

/*OpenCV 对所注册的鼠标回调函数定义为：
 void onMouse(int event, int x, int y, int flag, void *param)， 其中第四个参数 flag 为 event 下的附加状态，param 是用户传入的参数，我们都不需要使用，故不填写其参数名*/
void onMouse(int event, int x, int y, int, void*);

int main( int argc, char** argv) {
	/*		创建一个视频捕获对象，OpenCV 提供了一个 VideoCapture 对象，它屏蔽了从文件读取视频流和从摄像头读取摄像头的差异，
	当构造 函数参数为文件路径时，会从文件读取视频流；当构造函 数参数为设备编号时(第几个摄像头, 通常只有一个摄像头时为0)，会从摄像头处读取视频流。
	*/
	VideoCapture video("C:\\Users\\马春阳\\Videos\\Captures\\solar.mp4");    //读取文件，注意路径格式，VideoCapture video(0); 表示使用编号为0的摄像头

	namedWindow("Camshift");
	setMouseCallback("Camshift", onMouse, 0); // 1. 注册鼠标事件的回调函数, 第三个参数是用户提供给回调函数的，也就是回调函数中最后的 param 参数

	/*捕获画面的容器，OpenCV 中的 Mat 对象 OpenCV 中最关键的 Mat 类，Mat 是 Matrix(矩阵) 的缩写，OpenCV 中延续了像素图的概念，用矩阵来描述由像素构成的图像。*/
	Mat frame, hsv, hue, mask, hist, backproj;

	Rect trackWindow;   //追踪到窗口

	//30-32 计算直方图所必备的内容
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;

	while (true) {   
		video >> frame; //将 video 中的内容写入到 frame 中，这里 >> 运算符是经过 OpenCV 重载的

		if (frame.empty()) {  //当没有帧可继续读取时，退出循环
			break;
		}

		frame.copyTo(image); //将frame 中的图像写入全局变量 image 作为进行 Camshift 的缓存

		cvtColor(image, hsv, COLOR_BGR2HSV);  //转换到HSV空间

		if (trackObject) {   //当有目标时开始处理
			inRange(hsv, Scalar(0, 30, 10), Scalar(180, 256, 256), mask);  //只处理像素值为H：0~180，S：30~256，V：10~256之间的部分，过滤掉其他的部分并复制给 mask
			//48-50将 hsv 图像中的 H 通道分离出来
			int ch[] = { 0,0 };
			hue.create(hsv.size(), hsv.depth());
			mixChannels(&hsv, 1, &hue, 1, ch, 1);

			if (trackObject < 0) {  //如果需要追踪的物体还没有进行属性提取，则对选择的目标中的图像属性提取
				Mat roi(hue, selection), maskroi(mask, selection);   // 设置 H 通道和 mask 图像的 ROI
				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);   // 计算 ROI所在区域的直方图
				normalize(hist, hist, 0, 255, CV_MINMAX);    // 将直方图归一
				trackWindow = selection;		//设置追踪的窗口
				trackObject = 1;		//标记追踪的目标已经计算过直方图属性
			}
			calcBackProject(&hue, 1, 0, hist, backproj, &phranges);   //将直方图进行反向投影
			backproj &= mask;   //取公共部分

			//调用 Camshift 算法的接口
			RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));

			if (trackWindow.area() <= 1) {    //处理面积过小的情况
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r) & Rect(0, 0, cols, rows);
			}
			ellipse(image, trackBox, Scalar(0, 0, 255), 3, CV_AA); //绘制追踪区域
		}

		if (selectObject && selection.width > 0 && selection.height > 0) { // 如果正在选择追踪目标，则画出选择框
			Mat roi(image, selection);
			bitwise_not(roi, roi);   //对选择的区域图像反色
		}
		imshow("Camshift", image);   
		//录制视频帧率为 15, 等待 1000/15 保证视频播放流畅。waitKey(int delay) 是 OpenCV 提供的一个等待函数，当运行到这个函数时会阻塞 delay 毫秒的时间来等待键盘输入
		char c = (char)waitKey(1000 / 15.0);

		if (c == 27) {      //当按键为 ESC 时，退出循环
			break;
		}
	}

	destroyAllWindows;  // 释放申请的相关内存
	video.release();
	return 0;
}

void onMouse(int event, int x, int y, int, void*) {
	static Point origin;
	if (selectObject) {
		//确定鼠标选定区域的左上角坐标以及区域的长和宽
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = abs(x - origin.x);
		selection.height = abs(y - origin.y);
		selection &= Rect(0, 0, image.cols, image.rows); //& 运算符被 Rect 重载，表示两个区域取交集, 主要目的是为了处理当鼠标在选择区域时移除画面外
	}

	switch (event) {
		case CV_EVENT_LBUTTONDOWN:   //处理鼠标左键被按下
			origin = Point(x, y);
			selection = Rect(x, y, 0, 0);
			selectObject = true;
			break;
		case CV_EVENT_LBUTTONUP:   //处理鼠标左键被抬起
			selectObject = false;
			if (selection.width > 0 && selection.height > 0) {
				trackObject = -1; // 追踪的目标还未计算 Camshift 所需要的属性
			}
			break;
	}
}