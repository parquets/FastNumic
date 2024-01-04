#pragma once

namespace fastnum {
namespace cpu {

void softmax_last_axis(float* dest,
                       const float* source,
                       int in_channels,
                       int left_axis, int last_axis, 
                       bool logsoftmax=false);

void softmax_any_axis(float* dest,
                      const float* source,
                      int in_channels,
                      int left_axis, int mid_axis, int right_axis,
                      bool logsoftmax=false);

}
}