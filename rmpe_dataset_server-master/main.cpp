#include <iostream>
#include <vector>
#include "DataTransformer.h"
#include "H5Cpp.h"
#include "Eigen/Dense"
#include <zmq.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

struct DatumMetadata {
  uchar* datum;
  hsize_t shape[3];
};

int getKeys(std::vector<string> *keys, const H5::Group *group) {
  hsize_t  num_obj;
  H5Gget_num_objs(group->getId(), &num_obj);
  char buff[10];

  for (int i=0; i < num_obj; i++) {
    snprintf(buff, sizeof(buff), "%07d", i);
    keys->push_back(string(buff));
  }
  return 0;
}

int getDatum(const string key, const H5::Group *group, DatumMetadata &meta) {

  H5::DataSet *dataset = new H5::DataSet( group->openDataSet( key.data()));
  H5::DataSpace dataspace = dataset->getSpace();
  int rank = dataspace.getSimpleExtentNdims();
  dataspace.getSimpleExtentDims( meta.shape, NULL);
  H5::DataSpace mspace1(rank, meta.shape);

  meta.datum = new uchar[meta.shape[0] * meta.shape[1] * meta.shape[2]];

  dataset->read(meta.datum, H5::PredType::NATIVE_UCHAR, mspace1, dataspace);

  dataset->close();
  delete dataset;

  return 0;
}

int sendTransformed(
    uchar *transformed_data, double *transformed_label,
    const int rows_data,
    const int cols_data,
    const int rows_label,
    const int cols_label,
    const int ch,
    zmq::socket_t *s,
    int stop) {

  if (stop==1) {
    string headers = "{\"stop\":true}";
    zmq::message_t request (headers.size());
    memcpy (request.data (), (headers.c_str()), (headers.size()));
    s->send (request);
  } else {

    // prepare data
    Eigen::MatrixXd label = Eigen::Map<Eigen::MatrixXd>(
      transformed_label, rows_label * ch, cols_label);

    /* original
    Eigen::MatrixXd weights = Eigen::Map<Eigen::MatrixXd>(
        transformed_label, rows_label * np, cols_label);

    Eigen::MatrixXd vec = Eigen::Map<Eigen::MatrixXd>(
        transformed_label + start_label_data, rows_label * np, cols_label);

    Eigen::MatrixXd label = vec.cwiseProduct(weights);

    Eigen::MatrixXd mask = Eigen::Map<Eigen::MatrixXd>(
        transformed_label, rows_label, cols_label);*/

    const int lbl_rows(rows_label);
    const int lbl_cols(cols_label);
    const int dta_rows(rows_data);
    const int dta_cols(cols_data);

    // send headers
    string header_data =  "{\"descr\":\"uint8\",\"shape\":\"(3, 368,368)\",\"fortran_order\":false}";
    string header_label = "{\"descr\":\"<f8\",\"shape\":\"(41,46,46)\",\"fortran_order\":false}";
    string headers = "[" + header_data + "," + header_label + "]";
    zmq::message_t request (headers.size());
    memcpy (request.data (), (headers.c_str()), (headers.size()));
    s->send (request, ZMQ_SNDMORE);

    // send data
    size_t data_size(dta_rows * dta_cols * 3);
    zmq::message_t data_buff (transformed_data, data_size);
    s->send(data_buff, ZMQ_SNDMORE);

    // send label
    size_t label_size(lbl_rows * lbl_cols * sizeof(double) * ch);
    zmq::message_t lbl_buff (label.data(), label_size);
    s->send(lbl_buff);
  }

  return 0;
}

int main(int argc, char* argv[]) {

  // Check the number of parameters
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <INPUT DATASET>" << " <PORT>" << std::endl;
    return 1;
  }

  const string in_dataset = argv[1];
  const string port = argv[2];
  //const string batch_size = argv[3];
  //const int batch = atoi(batch_size.c_str());

  // Initialize zmq socket

  zmq::context_t ctx (1);
  zmq::socket_t s (ctx, ZMQ_PUSH);
  s.setsockopt(ZMQ_SNDHWM, 160);
  string bind_addr = "tcp://*:";
  bind_addr.append(port);
  int rc = zmq_bind (s, bind_addr.c_str());
  assert (rc == 0);
  int stop = 0;

  // Initialize params

  TransformationParameter params;

  params.stride=8;
  params.crop_size_x=368;
  params.crop_size_y=368;
  params.target_dist=0.6;
  params.scale_prob=1;
  params.scale_min=0.95;
  params.scale_max=1.15; // 偏大一点
  params.max_rotate_degree=15;
  params.center_perterb_max=5; //　尽量小一点
  params.do_clahe=false;
  params.visualize=false;
  params.num_parts=14;
  params.mirror = true;
  params.visualize = 0;
  params.num_limbs = 13;

  CPMDataTransformer* cpmDataTransformer = new CPMDataTransformer(params);
  cpmDataTransformer->InitRand();

  const int stride = params.stride;
  const int grid_x = params.crop_size_x / stride;
  const int grid_y = params.crop_size_y / stride;
  const int channelOffset = grid_y * grid_x;
  const int vec_channels = params.num_limbs * 2;
  const int heat_channels = params.num_parts + 1;  // add one bkg channel
  const int ch = vec_channels + heat_channels;

  uchar* transformed_data = new uchar[params.crop_size_x * params.crop_size_y * 3];
  double* transformed_label = new double[grid_x * grid_y * ch];

  // read all samples ids
  std::vector<string> keys;
  H5::H5File *f_in = new H5::H5File( in_dataset.c_str(), H5F_ACC_RDONLY );
  H5::Group* datum = new H5::Group( f_in->openGroup( "datum" ));

  getKeys(&keys, datum);

  cout << "Total samples: " << keys.size() << endl;

  int epoch_cnt = 0;
  int samples = keys.size();

  // process all samples for multiple epochs until stopped Ctrl+C
  // samples are shuffled for each epoch

  while (true) {

    cout << "Epoch: " << ++epoch_cnt << endl;

    std::random_shuffle(std::begin(keys), std::end(keys));

    // transform samples in current epoch
    for (int i = 0; i < keys.size(); i++) {
      string key = keys[i];

      if (i % 500 == 0) {
        cout << "curr_sample/total_samples/curr_epoch = " << i+1 << "/" << samples << "/" << epoch_cnt << endl;
      }
      // read sample

      DatumMetadata meta;
      getDatum(key, datum, meta);

      // transform sample

      int channels = meta.shape[0];
      int height = meta.shape[1];
      int width = meta.shape[2];

      double transformed_label0[grid_x * grid_y * ch] = {0};
      cpmDataTransformer->Transform_nv(meta.datum, channels, height, width,
                                       transformed_data, transformed_label0);

      // send transformed data and label to the socket

      sendTransformed(
          transformed_data,
          transformed_label0,
          params.crop_size_y,
          params.crop_size_x,
          grid_y,
          grid_x,
          ch,
          &s,
          stop);

      delete[] meta.datum;
    }
  }
}



