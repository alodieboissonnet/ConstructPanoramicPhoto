#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat I1 = imread("../IMG_0045.JPG", IMREAD_GRAYSCALE);
	Mat I2 = imread("../IMG_0046.JPG", IMREAD_GRAYSCALE);

	imshow("I1", I1);
	imshow("I2", I2);

	Ptr<AKAZE> D = AKAZE::create();
	vector<KeyPoint> m1, m2;
	Mat d1, d2;
	D->detectAndCompute(I1, noArray(), m1, d1);
	D->detectAndCompute(I2, noArray(), m2, d2);


	Mat J;
	drawKeypoints(I1, m1, J);
	//imshow("J", J);

	BFMatcher M(NORM_HAMMING);
	vector< vector<DMatch> > bfm_matches;
	M.knnMatch(d1, d2, bfm_matches, 2);

	drawMatches(I1, m1, I2, m2, bfm_matches, J);
	imshow("J", J);

	vector<KeyPoint> keypoints1, keypoints2;
	for(size_t i = 0; i < bfm_matches.size(); i++) {
		keypoints1.push_back(m1[bfm_matches[i][0].queryIdx]);
		keypoints2.push_back(m2[bfm_matches[i][0].trainIdx]);
	}

	vector<Point2f> m1points, m2points;
	KeyPoint::convert(keypoints1, m1points);
	KeyPoint::convert(keypoints2, m2points);

	Mat H = findHomography(m1points, m2points, RANSAC);

	Mat K(2 * I1.cols, I1.rows, CV_8U);
	warpPerspective(I2, I2, H, I2.size());
	hconcat(I1, I2, K);
	imshow("K", K);

	waitKey(0);
	return 0;
}
