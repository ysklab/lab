#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/viz/widgets.hpp>

using namespace std;
using namespace cv;

// load a ply file
// http://graphics.stanford.edu/data/3Dscanrep/
Mat cvcloud_load() {
  Mat cloud(1, 1889, CV_32FC3);
  ifstream ifs("/home/shang/Desktop/bunny.ply");

  string str;
  for (size_t i = 0; i < 12; ++i)
    getline(ifs, str);

  Point3f *data = cloud.ptr<cv::Point3f>();
  float dummy1, dummy2;
  for (size_t i = 0; i < 1889; ++i)
    ifs >> data[i].x >> data[i].y >> data[i].z >> dummy1 >> dummy2;

  cloud *= 5.0f;
  return cloud;
}

int main() {
  // step 1. construct window
  viz::Viz3d window("mywindow");
  window.showWidget("Coordinate Widget", viz::WCoordinateSystem());
  window.showWidget("xx", viz::WText("MMM", Point(10, 10), 40));

  // step 2. set the camera pose
  Vec3f cam_position(3.0f, 3.0f, -3.0f), cam_focal_point(3.f, 3.f, -4.0f),
      cam_y_direc(-1.0f, 0.0f, 0.0f);
  Affine3f cam_pose =
      viz::makeCameraPose(cam_position, cam_focal_point, cam_y_direc);
  Affine3f transform = viz::makeTransformToGlobal(
      Vec3f(0.0f, -1.0f, 0.0f), Vec3f(-1.0f, 0.0f, 0.0f),
      Vec3f(0.0f, 0.0f, -1.0f), cam_position);

  Mat bunny = cvcloud_load();
  viz::WCloud bunny_cloud(bunny, viz::Color::green());

  double z = 0.0f;
  Affine3f cloud_pose_global;
  while (!window.wasStopped()) {
    z += CV_PI * 0.01f;
    cloud_pose_global = transform.inv() *
                        Affine3f(Vec3f(0.0, 0.0, z), Vec3f(0.0, 0.0, 2.0)) *
                        Affine3f::Identity();
    window.showWidget("bunny_cloud", bunny_cloud, cloud_pose_global);

    // step 3. To show camera and frustum by pose
    // scale is 0.5
    viz::WCameraPosition camera(0.5);
    // show the frustum by intrinsic matrix
    viz::WCameraPosition camera_frustum(
        Matx33f(3.1, 0, 0.1, 0, 3.2, 0.2, 0, 0, 1));
    window.showWidget("Camera", camera, cam_pose);
    window.showWidget("Camera_frustum", camera_frustum, cam_pose);
    window.spinOnce(1, true);
  }
  return 0;
}