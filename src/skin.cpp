#include <skin.hpp>


//string s1 = "../imgs/";
//string s2 = ".jpg";


int cal(int x) {//曲线蒙版的曲线函数

	if (x < 127) return (int)(1.0 * 159 / 127 * x);
	else return (int)(1.0 * 96 / 128 * (x - 127) + 159);

}

Mat ChangeFacecolor(Mat img) {
	Mat img_ori = img.clone();//先复制原彩色图
	Mat c[3];//通道数组
	split(img, c);
	Mat img_gaussian;
	blur(c[0], img_gaussian, Size(8, 8));
	c[0] = c[0] - img_gaussian + 127;//对蓝色通道高反差保留
	Mat img_b = c[0];
	for (int i = 0; i < 3; i++)//对高反差保留后的蓝色通道做三次强光
	{
		for (int y = 0; y < img.rows; y++) {

			for (int x = 0; x < img.cols; x++) {

				int r = (int)img_b.at<uchar>(y, x);
				if (r > 127.5) {
					r = r + (255 - r)*(r - 127.5) / 127.5;

				}
				else {
					r = r * r / 127.5;

				}
				r = r > 255 ? 255 : r;
				r = r < 0 ? 0 : r;
				img_b.at<uchar>(y, x) = (int)r;
			}
		}
	}



		for (int y = 0; y < img.rows; y++) {

			for (int x = 0; x < img.cols; x++) {

				int r = (int)img_b.at<uchar>(y, x);
				int t = (int)c[0].at<uchar>(y, x);
				if (t < 220) {
					img.at<Vec3b>(y, x)[0] = cal(img.at<Vec3b>(y, x)[0]);//把暗色部分用曲线函数调亮
					img.at<Vec3b>(y, x)[1] = cal(img.at<Vec3b>(y, x)[1]);
					img.at<Vec3b>(y, x)[2] = cal(img.at<Vec3b>(y, x)[2]);
				}
							   			}
		}

	Mat out;
	bilateralFilter(img_ori, out, 40, 25 * 2, 25 / 2);//对原图进行双边滤波
	split(img, c);
	blur(c[2], img_gaussian, Size(1, 1));
	c[2] = c[2] - img_gaussian + 127;//对红色通道进行高反差保留
	Mat img_r = c[2];
	for (int y = 0; y < img.rows; y++) {

		for (int x = 0; x < img.cols; x++) {


			int r = int(img_r.at<uchar>(y, x));

			img.at<Vec3b>(y, x)[0] = img.at<Vec3b>(y, x)[0] + 2 * r - 255;//线性光混合
			img.at<Vec3b>(y, x)[1] = img.at<Vec3b>(y, x)[1] + 2 * r - 255;
			img.at<Vec3b>(y, x)[2] = img.at<Vec3b>(y, x)[2] + 2 * r - 255;



		}
	}

	img = out * 0.65 + img * 0.35;//双边滤波图透明度为0.65


	return img;




}


//int main(int argc, char **argv)
//{

//	Mat  img = imread(s1 + argv[1], 1);//读取原彩色图  
//	
//	img = ChangeFacecolor(img);//美白磨皮函数
//	
//	imwrite(s1 + "test_out.jpg", img);
//	waitKey();


//}
