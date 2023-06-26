#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/console/time.h>

bool IsGround(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, pcl::ModelCoefficients::Ptr coeff) {
    return coeff->values[2] > 0.80;
}
bool IsWall(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, pcl::ModelCoefficients::Ptr coeff) {
    return coeff->values[2] < 0.20;
}
int main () {
    pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2), cloud_filtered_blob (new pcl::PCLPointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the cloud data
    pcl::PCDReader reader;
    reader.read ("./data/my_scene.pcd", *cloud_blob);

    std::cerr << "PointCloud before filtering: " << cloud_blob->width * cloud_blob->height << " data points." << std::endl;

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud (cloud_blob);
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    sor.filter (*cloud_filtered_blob);

    // Convert to the templated PointCloud
    pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

    // Write the downsampled version to disk
    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZ> ("my_scene_downsampled.pcd", *cloud_filtered, false);

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.01);

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    int i = 0, nr_points = (int) cloud_filtered->size ();
    // While 30% of the original cloud is still there
    pcl::console::TicToc tt;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_picked(new pcl::PointCloud<pcl::PointXYZ>), cloud_rest(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZI>::Ptr ground_pc(new pcl::PointCloud<pcl::PointXYZI>), wall_pc(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ModelCoefficients::Ptr ground_coeff, wall_coeff;
    while (cloud_filtered->size () > 0.3 * nr_points) {
        // Segment the largest planar component from the remaining cloud
        tt.tic();
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        std::cout << " time of segment: " << tt.toc() << "ms" << std::endl;
        if (inliers->indices.size () == 0) {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        } else if (inliers->indices.size() < 0.2 * nr_points) {
            std::cerr << "Too few: " << inliers->indices.size() << " , sum=" << nr_points << std::endl;
            continue;
        }

        // Extract the inliers
        tt.tic();
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);
        extract.filter (*cloud_picked);
        std::cout << " time of extract: " << tt.toc() << "ms" << std::endl;
        if (IsGround(cloud_picked, coefficients)) {
            ground_coeff = coefficients;
            ground_pc->resize(cloud_picked->size());
            for (int i = 0; i < cloud_picked->size(); ++i) {
                ground_pc->at(i).x = cloud_picked->at(i).x;
                ground_pc->at(i).y = cloud_picked->at(i).y;
                ground_pc->at(i).z = cloud_picked->at(i).z;
                ground_pc->at(i).intensity = 10;
            }
        } else if (IsWall(cloud_picked, coefficients)) {
            wall_coeff = coefficients;
            wall_pc->resize(cloud_picked->size());
            for (int i = 0; i < cloud_picked->size(); ++i) {
                wall_pc->at(i).x = cloud_picked->at(i).x;
                wall_pc->at(i).y = cloud_picked->at(i).y;
                wall_pc->at(i).z = cloud_picked->at(i).z;
                wall_pc->at(i).intensity = 20;
            }
        }
        std::cerr << "planar component: " << cloud_picked->width * cloud_picked->height << " points, coeff=" << *coefficients << std::endl;

        std::string file = "my_scene_plane_" + std::to_string(i) + ".pcd";
        writer.write<pcl::PointXYZ> (file, *cloud_picked, false);

        // Create the filtering object
        extract.setNegative(true);
        extract.filter(*cloud_rest);
        cloud_filtered.swap(cloud_rest);
        i++;
    }
    std::vector<float> v(4);
    if (ground_coeff && wall_pc) {

    } else if (ground_coeff && !wall_pc) {  // only ground
        v = ground_coeff->values;
    } else {  // only wall

    }

    return (0);
}