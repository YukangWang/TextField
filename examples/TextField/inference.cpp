#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <time.h>
#include <queue>

#define PI 3.14159265

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

struct coord {
    int x;
    int y;
};

int main(int argc, char** argv) {
  if (argc != 9) {
    cerr << "Usage: " << argv[0] << " deploy.prototxt network.caffemodel gpu input h w thr output" << endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string net_file = argv[1];
  string weight_file = argv[2];

  Caffe::SetDevice(atoi(argv[3]));
  Caffe::set_mode(Caffe::GPU);

  shared_ptr<Net<float> > net_;
  net_.reset(new Net<float>(net_file, TEST));
  net_->CopyTrainedLayersFrom(weight_file);

  string input_dir = argv[4];
  string input_file = input_dir+"*.jpg";
  vector<String> fn;
  glob(input_file, fn, false);
  int fn_count = fn.size();

  int height = atoi(argv[5]);
  int width = atoi(argv[6]);
  float thr = atof(argv[7]);
  string output_dir = argv[8];

  clock_t start_time = clock();
  float p_sum = 0;

  for (int idx = 0; idx < fn_count; idx++) {
    cout << fn[idx] << endl;
    // cout << fn[idx].substr(fn[i].rfind("/")+1) << endl;
    Mat img = imread(fn[idx], 1);
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    int channel = img.channels();
    // height = raw_h;
    // width = raw_w;
    resize(img, img, Size(width, height));

    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, channel, height, width);
    net_->Reshape();
    // cout << input_layer->height() << input_layer->width() << input_layer->channels() << endl;
  
    float* input_data = input_layer->mutable_cpu_data();
    vector<Mat> input_channels;
    for (int i = 0; i < input_layer->channels(); ++i) {
      Mat channel(height, width, CV_32FC1, input_data);
      input_channels.push_back(channel);
      input_data += height * width;
    }
  
    // Mat mean_ = Mat(raw_h, raw_w, CV_32FC3);
    Mat mean_ = Mat(height, width, CV_32FC3);
    mean_ = Scalar(103.939, 116.779, 123.68);
    img.convertTo(img, CV_32FC3);
    subtract(img, mean_, img);
    // resize(img, img, Size(width, height));
    split(img, input_channels);
  
    net_->Forward();
  
    Blob<float>* output_layer = net_->output_blobs()[0];
    float* output_data = output_layer->mutable_cpu_data();
    // cout << output_layer->height() << output_layer->width() << output_layer->channels() << endl;
    Mat predx(height, width, CV_32FC1, output_data);
    output_data += height * width;
    Mat predy(height, width, CV_32FC1, output_data);
  
    clock_t p_start = clock();
  
    Mat magnitude(height, width, CV_32FC1);
    Mat angle(height, width, CV_32FC1);
    cartToPolar(predx, predy, magnitude, angle);
  
    // double v_min, v_max;
    // double a_min, a_max;
    // minMaxIdx(magnitude, &v_min, &v_max);
    // minMaxIdx(angle, &a_min, &a_max);
  
    Mat mask;
    compare(magnitude, Scalar(thr), mask, CMP_GT);
    mask.convertTo(mask, CV_32FC1);

    Mat parent(height, width, CV_32FC2, Scalar(0,0));
    Mat ending(height, width, CV_32FC1, Scalar(0));
    Mat merged_ending(height, width, CV_32FC1, Scalar(0));
  
    for (int row = 0; row < height; row++) {
      float* mask_p = mask.ptr<float>(row);
      float* angle_p = angle.ptr<float>(row);
      float* parent_p = parent.ptr<float>(row);
      float* ending_p = ending.ptr<float>(row);
      for (int col = 0; col < width; col++) {
        if (mask_p[col] == 255) {
          if (angle_p[col] < PI/8 || angle_p[col] >= 15*PI/8) {
            parent_p[2*col] = 1;
            parent_p[2*col+1] = 0;
            if (row+1 <= height-1) {
              float* mask_pn = mask.ptr<float>(row+1);
              if (mask_pn[col] == 0) ending_p[col] = 1;
            }
          }
          else if (angle_p[col] >= PI/8 && angle_p[col] < 3*PI/8) {
            parent_p[2*col] = 1;
            parent_p[2*col+1] = 1;
            if (row+1 <= height-1 && col+1 <= width-1) {
              float* mask_pn = mask.ptr<float>(row+1);
              if (mask_pn[col+1] == 0) ending_p[col] = 1;
            }
          }
          else if (angle_p[col] >= 3*PI/8 && angle_p[col] < 5*PI/8) {
            parent_p[2*col] = 0;
            parent_p[2*col+1] = 1;
            if (col+1 <= width-1) {
              float* mask_pn = mask.ptr<float>(row);
              if (mask_pn[col+1] == 0) ending_p[col] = 1;
            }
          }
          else if (angle_p[col] >= 5*PI/8 && angle_p[col] < 7*PI/8) {
            parent_p[2*col] = -1;
            parent_p[2*col+1] = 1;
            if (row-1 >= 0 && col+1 <= width-1) {
              float* mask_pn = mask.ptr<float>(row-1);
              if (mask_pn[col+1] == 0) ending_p[col] = 1;
            }
          }
          else if (angle_p[col] >= 7*PI/8 && angle_p[col] < 9*PI/8) {
            parent_p[2*col] = -1;
            parent_p[2*col+1] = 0;
            if (row-1 >= 0) {
              float* mask_pn = mask.ptr<float>(row-1);
              if (mask_pn[col] == 0) ending_p[col] = 1;
            }
          }
          else if (angle_p[col] >= 9*PI/8 && angle_p[col] < 11*PI/8) {
            parent_p[2*col] = -1;
            parent_p[2*col+1] = -1;
            if (row-1 >= 0 && col-1 >= 0) {
              float* mask_pn = mask.ptr<float>(row-1);
              if (mask_pn[col-1] == 0) ending_p[col] = 1;
            }
          }
          else if (angle_p[col] >= 11*PI/8 && angle_p[col] < 13*PI/8) {
            parent_p[2*col] = 0;
            parent_p[2*col+1] = -1;
            if (col-1 >= 0) {
              float* mask_pn = mask.ptr<float>(row);
              if (mask_pn[col-1] == 0) ending_p[col] = 1;
            }
          }
          else if (angle_p[col] >= 13*PI/8 && angle_p[col] < 15*PI/8) {
            parent_p[2*col] = 1;
            parent_p[2*col+1] = -1;
            if (row+1 <= height-1 && col-1 >= 0) {
              float* mask_pn = mask.ptr<float>(row+1);
              if (mask_pn[col-1] == 0) ending_p[col] = 1;
            }
          }
        }
      }
    }
  
    coord p, pc, pt;
    Mat visited(height, width, CV_32FC1, Scalar(0));
    Mat dict(height, width, CV_32FC2, Scalar(0,0));
  
    int sup_idx = 1;
    for (int row = 0; row < height; row++) {
      float* mask_p = mask.ptr<float>(row);
      float* visited_p = visited.ptr<float>(row);
      for (int col = 0; col < width; col++) {
        if (mask_p[col] == 255 && !visited_p[col]) {
          p.x = row;
          p.y = col;
          queue<coord> Q;
          Q.push(p);
          while (!Q.empty()) {
            pc = Q.front();
            float* parent_pc = parent.ptr<float>(pc.x);
            float* visited_pc = visited.ptr<float>(pc.x);
            float* dict_pc = dict.ptr<float>(pc.x);
            dict_pc[2*pc.y] = sup_idx;
            visited_pc[pc.y] = 1;
            for (int dx = -1; dx <= 1; dx++) {
              for (int dy = -1; dy <= 1; dy++) {
                pt.x = pc.x + dx;
                pt.y = pc.y + dy;
                if (pt.x >= 0 && pt.x <= height-1 && pt.y >= 0 && pt.y <= width-1) {
                  float* parent_pt = parent.ptr<float>(pt.x);
                  float* visited_pt = visited.ptr<float>(pt.x);
                  float* dict_pt = dict.ptr<float>(pt.x);
                  if (!visited_pt[pt.y] && (parent_pt[2*pt.y] != 0 || parent_pt[2*pt.y+1] != 0)) {
                    if (parent_pt[2*pt.y] == -1*dx && parent_pt[2*pt.y+1] == -1*dy) {
                      Q.push(pt);
                      dict_pc[2*pc.y+1] = max(dict_pc[2*pc.y+1], dict_pt[2*pt.y+1]+1);
                    }
                    else if (parent_pc[2*pc.y] == 1*dx && parent_pc[2*pc.y+1] == 1*dy) {
                      Q.push(pt);
                      dict_pt[2*pt.y+1] = max(dict_pt[2*pt.y+1], dict_pc[2*pc.y+1]+1);
                    }
                  }
                }
              }
            }
            Q.pop();
          }
          sup_idx++;
        }
      }
    }
  
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(ending, merged_ending, element);
  
    for (int row = 0; row < height; row++) {
      float* ending_p = ending.ptr<float>(row);
      float* parent_p = parent.ptr<float>(row);
      float* dict_p = dict.ptr<float>(row);
      for (int col = 0; col < width; col++) {
        if (ending_p[col] == 1) {
          for (int dilDepth = 1; dilDepth <= min((int)(1*dict_p[2*col+1]-16), 12); dilDepth++) {
            p.x = row+(int)parent_p[2*col]*dilDepth;
            p.y = col+(int)parent_p[2*col+1]*dilDepth;
            if (p.x >= 0 && p.x <= height-1 && pt.y >= 0 && pt.y <= width-1) {
              float* merged_ending_p = merged_ending.ptr<float>(p.x);
              merged_ending_p[p.y] = 1;
            }
          }
        }
      }
    }
  
    Mat cctmp, label;
    merged_ending.convertTo(cctmp, CV_8U);
    int ccnum = connectedComponents(cctmp, cctmp, 8, CV_16U);
    cctmp.convertTo(label, CV_32F);
  
    int sup_map_cc[sup_idx] = {0};
    int stat[ccnum][8] = {0};
    for (int row = 0; row < height; row++) {
      float* ending_p = ending.ptr<float>(row);
      float* parent_p = parent.ptr<float>(row);
      float* label_p = label.ptr<float>(row);
      float* dict_p = dict.ptr<float>(row);
      for (int col = 0; col < width; col++) {
        if (ending_p[col] == 1) {
          int dx = (int)parent_p[2*col];
          int dy = (int)parent_p[2*col+1];
          int cc_idx = (int)label_p[col];
          sup_map_cc[(int)dict_p[2*col]] = cc_idx;
          if (dx == 1 && dy == 0) stat[cc_idx][0]++;
          if (dx == 1 && dy == 1) stat[cc_idx][1]++;
          if (dx == 0 && dy == 1) stat[cc_idx][2]++;
          if (dx == -1 && dy == 1) stat[cc_idx][3]++;
          if (dx == -1 && dy == 0) stat[cc_idx][4]++;
          if (dx == -1 && dy == -1) stat[cc_idx][5]++;
          if (dx == 0 && dy == -1) stat[cc_idx][6]++;
          if (dx == 1 && dy == -1) stat[cc_idx][7]++;
        }
      }
    }
    int cc_map_filted[ccnum] = {0};
    int filted_idx = 1;
    for (int cc_idx = 1; cc_idx <= ccnum-1; cc_idx++) {
      int dif1 = max(stat[cc_idx][0], stat[cc_idx][4]) - min(stat[cc_idx][0], stat[cc_idx][4]);
      int dif2 = max(stat[cc_idx][1], stat[cc_idx][5]) - min(stat[cc_idx][1], stat[cc_idx][5]);
      int dif3 = max(stat[cc_idx][2], stat[cc_idx][6]) - min(stat[cc_idx][2], stat[cc_idx][6]);
      int dif4 = max(stat[cc_idx][3], stat[cc_idx][7]) - min(stat[cc_idx][3], stat[cc_idx][7]);
      int sum1 = stat[cc_idx][0]+stat[cc_idx][1]+stat[cc_idx][2]+stat[cc_idx][3];
      int sum2 = stat[cc_idx][4]+stat[cc_idx][5]+stat[cc_idx][6]+stat[cc_idx][7];
      int difsum = max(sum1, sum2) - min(sum1, sum2);
      int sum = sum1 + sum2;
      float ratio1 = (float)(difsum) / (float)sum;
      float ratio2 = (float)(dif1+dif2+dif3+dif4) / (float)sum;
      if (ratio1 <= 0.6 && ratio2 <= 0.6) {
      //  cout<<ratio1<<","<<ratio2<<","<<sum<<endl;
        cc_map_filted[cc_idx] = filted_idx;
        filted_idx++;
      }
    }
  
    for (int row = 0; row < height; row++) {
      float* dict_p = dict.ptr<float>(row);
      float* label_p = label.ptr<float>(row);
      for (int col = 0; col < width; col++) {
        if (label_p[col] == 0) {
          label_p[col] = cc_map_filted[(int)sup_map_cc[(int)dict_p[2*col]]];
        }
        else {
          label_p[col] = cc_map_filted[(int)label_p[col]];
        }
      }
    }
  
    Mat clstmp;
    Mat res(height, width, CV_32FC1, Scalar(0));
    Mat element_ = getStructuringElement(MORPH_RECT, Size(11, 11));
    for (int i = 1; i < filted_idx; i++) {
      compare(label, Scalar(i), clstmp, CMP_EQ);
      dilate(clstmp, clstmp, element_);
      erode(clstmp, clstmp, element_);
      compare(clstmp, Scalar(0), clstmp, CMP_GT);
      clstmp.convertTo(clstmp, CV_32FC1);
      multiply(res, 1-clstmp/255, res);
      add(res, clstmp/255*i, res);
    }
    resize(res, res, Size(raw_w, raw_h), 0, 0, INTER_NEAREST);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PXM_BINARY);
    // compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    // compression_params.push_back(2);
    string output_fn = output_dir+fn[idx].substr(0,fn[idx].length()-4).substr(fn[idx].rfind("/")+1)+".pgm";
    // string output_fn = output_dir+fn[idx].substr(0,fn[idx].length()-4).substr(fn[idx].rfind("/")+1)+".png";
    // cout << output_fn << endl;
    imwrite(output_fn, res, compression_params);
    clock_t p_end = clock();
    p_sum += static_cast<float>(p_end - p_start);
  }
  
  clock_t end_time = clock();
  cout<<"-----Runtime(all): "<<static_cast<float>(end_time - start_time)/CLOCKS_PER_SEC/fn_count<<" s/image-----"<<endl;
  cout<<"-----Runtime(post-processing): "<<p_sum/CLOCKS_PER_SEC/fn_count<<" s/image-----"<<endl;
}

#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
