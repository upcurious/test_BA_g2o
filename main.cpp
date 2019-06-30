
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/concept_check.hpp>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace Eigen;

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr ;
}

g2o::SE3Quat toSE3Quat(const Eigen::Isometry3d &Tcw)
{

    return g2o::SE3Quat(Tcw.linear(),Tcw.translation());
}
class FeaturePerFrame
{
  public:
    FeaturePerFrame(Vector2d _uv,double _depth):uv(_uv),depth(_depth){}
    Vector2d uv;
    double depth;
};
class Observation
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Observation ( int mpt_id,int _start_frame,Eigen::Vector3d _ptw)
        :mpt_id_ ( mpt_id ),start_frame(_start_frame),ptw(_ptw){}
    Eigen::Vector3d ptw;
    int mpt_id_;
    int start_frame;
    vector<pair<FeaturePerFrame,int>> cam_id_;
};
void createData ( int n_mappoints, int n_cameras, double fx, double fy, double cx, double cy,
                  double height, double width,
                  std::vector<Eigen::Vector3d>& mappoints, std::vector<Eigen::Isometry3d>& cameras,
                  std::vector<Observation>& observations )
{
    
    const double angle_range = 0.1;
    const double x_range = 1.0;
    const double y_range = 1.0;
    const double z_range = 0.5;

    
    const double x_min = -5.0;
    const double x_max = 5.0;
    const double y_min = -5.0;
    const double y_max = 5.0;
    const double z_min = 0.6;
    const double z_max = 8.0;

    cv::RNG rng ( cv::getTickCount() );

    
    Eigen::Matrix3d Rx, Ry, Rz;
    Eigen::Matrix3d R; 
    Eigen::Vector3d t;
    for ( int i = 0; i < n_cameras; i ++ ) {
        
        double tz = rng.uniform ( -angle_range, angle_range );
        double ty = rng.uniform ( -angle_range, angle_range );
        double tx = rng.uniform ( -angle_range, angle_range );

        Rz << cos ( tz ), -sin ( tz ), 0.0,
           sin ( tz ), cos ( tz ), 0.0,
           0.0, 0.0, 1.0;
        Ry << cos ( ty ), 0.0, sin ( ty ),
           0.0, 1.0, 0.0,
           -sin ( ty ), 0.0, cos ( ty );
        Rx << 1.0, 0.0, 0.0,
           0.0, cos ( tx ), -sin ( tx ),
           0.0, sin ( tx ), cos ( tx );
        R = Rz * Ry * Rx;

        
        double x = rng.uniform ( -x_range, x_range );
        double y = rng.uniform ( -y_range, y_range );
        double z = rng.uniform ( -z_range, z_range );
        t << x, y, z;

        Eigen::Isometry3d cam;
	cam.linear() = R;
	cam.translation() = t;
        cameras.push_back ( cam );
    } 


    std::vector<Eigen::Vector3d> tmp_mappoints;
    for ( int i = 0; i < n_mappoints; i ++ ) {
        double x = rng.uniform ( x_min, x_max );
        double y = rng.uniform ( y_min, y_max );
        double z = rng.uniform ( z_min, z_max );
        tmp_mappoints.push_back ( Eigen::Vector3d ( x,y,z ) );
    }

    
    for ( int i = 0; i < n_mappoints; i ++ ) {
        const Eigen::Vector3d& ptw = tmp_mappoints.at ( i );
        int n_obs = 0.0;
        for ( int nc = 0; nc < n_cameras; nc ++ ) {
            const Eigen::Isometry3d& cam_pose = cameras[nc];
            
            const Eigen::Vector3d ptc = cam_pose * ptw;
            Eigen::Vector2d uv (
                fx*ptc[0]/ptc[2] + cx,
                fy*ptc[1]/ptc[2] + cy
            );

            if ( uv[0]<0 || uv[1]<0 || uv[0]>=width || uv[1]>=height || ptc[2] < 0.1 ) {
                continue;
            }
            n_obs ++;
        }

        if ( n_obs < 2 ) {
            continue;
        }

        mappoints.push_back ( ptw );
    }

    for ( size_t i = 0; i < mappoints.size(); i ++ ) {
        const Eigen::Vector3d& ptw = mappoints.at ( i );
	Observation ob (i,0,ptw);
        for ( int nc = 0; nc < n_cameras; nc ++ ) {
            const Eigen::Isometry3d& cam_pose = cameras[nc];

            const Eigen::Vector3d ptc = cam_pose * ptw;
            Eigen::Vector2d uv (
                fx*ptc[0]/ptc[2] + cx,
                fy*ptc[1]/ptc[2] + cy
            );
	    
	    FeaturePerFrame obs(uv,ptc[2]);
	    ob.cam_id_.push_back(make_pair(obs,nc));
            
        }
        observations.push_back ( ob );
    }
    
    mappoints.shrink_to_fit();
    cameras.shrink_to_fit();
    observations.shrink_to_fit();
}

void addNoise ( std::vector< Eigen::Vector3d >& mappoints, std::vector< Eigen::Isometry3d >& cameras, std::vector< Observation >& observations, double mpt_noise, double cam_trans_noise, double cam_rot_noise, double ob_noise )
{
    cv::RNG rng ( cv::getTickCount() );

    for ( size_t i = 0; i < mappoints.size(); i ++ ) {
        double nx = rng.gaussian ( mpt_noise );
        double ny = rng.gaussian ( mpt_noise );
        double nz = rng.gaussian ( mpt_noise );
        mappoints.at ( i ) += Eigen::Vector3d ( nx, ny, nz );
    }

	Eigen::Matrix3d Rx, Ry, Rz;
	Eigen::Matrix3d R; 
	Eigen::Vector3d t;
	for(size_t i = 0; i < cameras.size(); i ++)
	{
		
		if(i == 0)
			continue;
		
		double tz = rng.gaussian ( cam_rot_noise );
		double ty = rng.gaussian ( cam_rot_noise );
		double tx = rng.gaussian ( cam_rot_noise );
		
		Rz << cos ( tz ), -sin ( tz ), 0.0,
		sin ( tz ), cos ( tz ), 0.0,
		0.0, 0.0, 1.0;
		Ry << cos ( ty ), 0.0, sin ( ty ),
		0.0, 1.0, 0.0,
		-sin ( ty ), 0.0, cos ( ty );
		Rx << 1.0, 0.0, 0.0,
		0.0, cos ( tx ), -sin ( tx ),
		0.0, sin ( tx ), cos ( tx );
		R = Rz * Ry * Rx;
		
		
		double x = rng.gaussian ( cam_trans_noise );
		double y = rng.gaussian ( cam_trans_noise );
		double z = rng.gaussian ( cam_trans_noise );
		t << x, y, z;
		
		
		Eigen::Isometry3d cam_noise;
		cam_noise.linear() = R;
		cam_noise.translation() = t;
		cameras[i] = cameras[i]*cam_noise;
	}

	for(auto &map : observations)
	  for(auto &per_it : map.cam_id_)
	{
		double x = rng.gaussian ( ob_noise );
		double y = rng.gaussian ( ob_noise );
		per_it.first.uv += Eigen::Vector2d(x,y);
	}
}


int main( int argc, char** argv )
{
    const int n_mappoints = 1000;
    const int n_cameras = 6;

    const double fx = 525.0;
    const double fy = 525.0;
    const double cx = 320.0;
    const double cy = 240.0;
    const double height = 640;
    const double width = 480;
	
    std::cout << "Start create data...\n";
    std::vector<Eigen::Vector3d> mappoints;
    std::vector<Eigen::Isometry3d> cameras;
    std::vector<Observation> observations;
    createData ( n_mappoints, n_cameras, fx, fy, cx, cy, height, width, mappoints, cameras, observations );
    std::cout << "Total mpt: " << mappoints.size() << "  cameras: " << cameras.size() << "  observations: " << observations.size() << std::endl;
    double mpt_noise = 0.01;
    double cam_trans_noise = 0.1;
    double cam_rot_noise = 0.1;
    double ob_noise = 1.0;
    
    std::vector<Eigen::Vector3d> noise_mappoints;
    noise_mappoints = mappoints;
    std::vector<Eigen::Isometry3d> noise_cameras;
    noise_cameras = cameras;
    std::vector<Observation> noise_observations;
    noise_observations = observations;
    addNoise(noise_mappoints, noise_cameras, noise_observations, mpt_noise, cam_trans_noise, cam_rot_noise, ob_noise );

    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();
    Block* solver_ptr = new Block( std::unique_ptr<Block::LinearSolverType>(linearSolver) );
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr) );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( algorithm );
    
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );
    for ( int i=0; i<noise_cameras.size(); i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        v->setFixed( i == 0 );
	g2o::SE3Quat se3 = toSE3Quat(noise_cameras[i]);
        v->setEstimate( se3 );
        optimizer.addVertex( v );
    }
    
    for ( auto &it_per_id :noise_observations)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
	
	const int id = it_per_id.mpt_id_+noise_cameras.size();
	v->setId(id);
        v->setMarginalized(true);
        v->setEstimate(it_per_id.ptw);
        optimizer.addVertex( v );
	for (auto &it_per_frame : it_per_id.cam_id_)
	{
	  Eigen::Matrix<double,2,1> obs = it_per_frame.first.uv;
	  g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
	  edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(id)) );
	  edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(it_per_frame.second)) );
	  edge->setMeasurement( obs );
	  edge->setInformation( Eigen::Matrix2d::Identity() );
	  edge->setParameterId(0, 0);
	  
	  edge->setRobustKernel( new g2o::RobustKernelHuber() );
	  optimizer.addEdge( edge );
	}
    }
   
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    
    std::vector<Eigen::Isometry3d> cam_Tw;
    
    for (int i = 0; i <= noise_cameras.size(); i++)
    {
	g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
	g2o::SE3Quat SE3quat = vSE3->estimate();
	
	cam_Tw.push_back(Isometry3d(SE3quat));
    }
    double sum_rot_error = 0.0;
    double sum_trans_error = 0.0;
    for(size_t i = 0; i < cameras.size(); i ++)
    {
	Eigen::Isometry3d opt_pose = noise_cameras[i];
	Eigen::Isometry3d org_pose = cameras.at(i);
	Eigen::Isometry3d pose_err = opt_pose * org_pose.inverse();
	sum_rot_error += R2ypr(pose_err.linear()).norm();
	sum_trans_error += pose_err.translation().norm();
    }
    std::cout << "Pre Mean rot error: " << sum_rot_error / (double)(cameras.size())
    << "\tPre Mean trans error: " <<  sum_trans_error / (double)(cameras.size()) << std::endl;
    
     sum_rot_error = 0.0;
     sum_trans_error = 0.0;
    for(size_t i = 0; i < cameras.size(); i ++)
    {
	    Eigen::Isometry3d opt_pose = cam_Tw[i];
	    Eigen::Isometry3d org_pose = cameras.at(i);
	    Eigen::Isometry3d pose_err = opt_pose * org_pose.inverse();
	    sum_rot_error += R2ypr(pose_err.linear()).norm();
	    sum_trans_error += pose_err.translation().norm();
    }
    std::cout << "Post Mean rot error: " << sum_rot_error / (double)(cameras.size())
    << "\tPost Mean trans error: " <<  sum_trans_error / (double)(cameras.size()) << std::endl;
    return 0;
}
