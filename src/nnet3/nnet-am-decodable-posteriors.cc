// nnet3/nnet-am-decodable-simple.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet3/nnet-am-decodable-posteriors.h"

namespace kaldi {
namespace nnet3 {

DecodableAmPosteriors::DecodableAmPosteriors(
    const MatrixBase<BaseFloat> &posteriors,
    const TransitionModel &trans_model):
    posteriors_copy_(NULL),
    trans_model_(trans_model) {
  posteriors_copy_ = new Matrix<BaseFloat>(posteriors);
}



BaseFloat DecodableAmPosteriors::LogLikelihood(int32 frame,
                                               int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdf(transition_id);
  return (*posteriors_copy_)(frame, pdf_id);
}

} // namespace nnet3
} // namespace kaldi
