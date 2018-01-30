#include <opencv2/opencv.hpp>   //OpenCVͷ�ļ�
#include <iostream>
using namespace cv;   //���� OpenCV���ƿռ�
using namespace std;

bool selectObject = false;  //���ڱ���Ƿ���ѡȡĿ��
int trackObject = 0; //1 ��ʾ��׷�ٶ��� 0 ��ʾ��׷�ٶ��� -1 ��ʾ׷�ٶ�����δ���� Camshift ���������
Rect selection;   //�������ѡ������
Mat image;     //�����ȡ������Ƶ֡

/*OpenCV ����ע������ص���������Ϊ��
 void onMouse(int event, int x, int y, int flag, void *param)�� ���е��ĸ����� flag Ϊ event �µĸ���״̬��param ���û�����Ĳ��������Ƕ�����Ҫʹ�ã��ʲ���д�������*/
void onMouse(int event, int x, int y, int, void*);

int main( int argc, char** argv) {
	/*		����һ����Ƶ�������OpenCV �ṩ��һ�� VideoCapture �����������˴��ļ���ȡ��Ƶ���ʹ�����ͷ��ȡ����ͷ�Ĳ��죬
	������ ��������Ϊ�ļ�·��ʱ������ļ���ȡ��Ƶ���������캯 ������Ϊ�豸���ʱ(�ڼ�������ͷ, ͨ��ֻ��һ������ͷʱΪ0)���������ͷ����ȡ��Ƶ����
	*/
	VideoCapture video("C:\\Users\\����\\Videos\\Captures\\solar.mp4");    //��ȡ�ļ���ע��·����ʽ��VideoCapture video(0); ��ʾʹ�ñ��Ϊ0������ͷ

	namedWindow("Camshift");
	setMouseCallback("Camshift", onMouse, 0); // 1. ע������¼��Ļص�����, �������������û��ṩ���ص������ģ�Ҳ���ǻص����������� param ����

	/*�������������OpenCV �е� Mat ���� OpenCV ����ؼ��� Mat �࣬Mat �� Matrix(����) ����д��OpenCV ������������ͼ�ĸ���þ��������������ع��ɵ�ͼ��*/
	Mat frame, hsv, hue, mask, hist, backproj;

	Rect trackWindow;   //׷�ٵ�����

	//30-32 ����ֱ��ͼ���ر�������
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;

	while (true) {   
		video >> frame; //�� video �е�����д�뵽 frame �У����� >> ������Ǿ��� OpenCV ���ص�

		if (frame.empty()) {  //��û��֡�ɼ�����ȡʱ���˳�ѭ��
			break;
		}

		frame.copyTo(image); //��frame �е�ͼ��д��ȫ�ֱ��� image ��Ϊ���� Camshift �Ļ���

		cvtColor(image, hsv, COLOR_BGR2HSV);  //ת����HSV�ռ�

		if (trackObject) {   //����Ŀ��ʱ��ʼ����
			inRange(hsv, Scalar(0, 30, 10), Scalar(180, 256, 256), mask);  //ֻ��������ֵΪH��0~180��S��30~256��V��10~256֮��Ĳ��֣����˵������Ĳ��ֲ����Ƹ� mask
			//48-50�� hsv ͼ���е� H ͨ���������
			int ch[] = { 0,0 };
			hue.create(hsv.size(), hsv.depth());
			mixChannels(&hsv, 1, &hue, 1, ch, 1);

			if (trackObject < 0) {  //�����Ҫ׷�ٵ����廹û�н���������ȡ�����ѡ���Ŀ���е�ͼ��������ȡ
				Mat roi(hue, selection), maskroi(mask, selection);   // ���� H ͨ���� mask ͼ��� ROI
				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);   // ���� ROI���������ֱ��ͼ
				normalize(hist, hist, 0, 255, CV_MINMAX);    // ��ֱ��ͼ��һ
				trackWindow = selection;		//����׷�ٵĴ���
				trackObject = 1;		//���׷�ٵ�Ŀ���Ѿ������ֱ��ͼ����
			}
			calcBackProject(&hue, 1, 0, hist, backproj, &phranges);   //��ֱ��ͼ���з���ͶӰ
			backproj &= mask;   //ȡ��������

			//���� Camshift �㷨�Ľӿ�
			RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));

			if (trackWindow.area() <= 1) {    //���������С�����
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r, trackWindow.x + r, trackWindow.y + r) & Rect(0, 0, cols, rows);
			}
			ellipse(image, trackBox, Scalar(0, 0, 255), 3, CV_AA); //����׷������
		}

		if (selectObject && selection.width > 0 && selection.height > 0) { // �������ѡ��׷��Ŀ�꣬�򻭳�ѡ���
			Mat roi(image, selection);
			bitwise_not(roi, roi);   //��ѡ�������ͼ��ɫ
		}
		imshow("Camshift", image);   
		//¼����Ƶ֡��Ϊ 15, �ȴ� 1000/15 ��֤��Ƶ����������waitKey(int delay) �� OpenCV �ṩ��һ���ȴ������������е��������ʱ������ delay �����ʱ�����ȴ���������
		char c = (char)waitKey(1000 / 15.0);

		if (c == 27) {      //������Ϊ ESC ʱ���˳�ѭ��
			break;
		}
	}

	destroyAllWindows;  // �ͷ����������ڴ�
	video.release();
	return 0;
}

void onMouse(int event, int x, int y, int, void*) {
	static Point origin;
	if (selectObject) {
		//ȷ�����ѡ����������Ͻ������Լ�����ĳ��Ϳ�
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = abs(x - origin.x);
		selection.height = abs(y - origin.y);
		selection &= Rect(0, 0, image.cols, image.rows); //& ������� Rect ���أ���ʾ��������ȡ����, ��ҪĿ����Ϊ�˴��������ѡ������ʱ�Ƴ�������
	}

	switch (event) {
		case CV_EVENT_LBUTTONDOWN:   //����������������
			origin = Point(x, y);
			selection = Rect(x, y, 0, 0);
			selectObject = true;
			break;
		case CV_EVENT_LBUTTONUP:   //������������̧��
			selectObject = false;
			if (selection.width > 0 && selection.height > 0) {
				trackObject = -1; // ׷�ٵ�Ŀ�껹δ���� Camshift ����Ҫ������
			}
			break;
	}
}