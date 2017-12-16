// nnet3/nnet-am-decodable-posterior.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_AM_DECODABLE_POSTERIORS_H_
#define KALDI_NNET3_NNET_AM_DECODABLE_POSTERIORS_H_

#include <vector>
#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"

namespace kaldi {
namespace nnet3 {

/*
 * Decoding with posteriors:
 *   - Constructor: Copy posteriors
 *   - LogLikelihood: lookup posterior
 */

struct NnetPosteriorOptions {
    int32 extra_left_context;
    int32 extra_right_context;
    int32 extra_left_context_initial;
    int32 extra_right_context_final;
    int32 frame_subsampling_factor;
    int32 frames_per_chunk;
    BaseFloat acoustic_scale;

    NnetPosteriorOptions():
        extra_left_context(0),
        extra_right_context(0),
        extra_left_context_initial(-1),
        extra_right_context_final(-1),
        frame_subsampling_factor(1),
        frames_per_chunk(50),
        acoustic_scale(0.1) {}

    void Register(OptionsItf *opts) {
      opts->Register("extra-left-context", &extra_left_context,
                     "Number of frames of additional left-context to add on top "
                         "of the neural net's inherent left context (may be useful in "
                         "recurrent setups");
      opts->Register("extra-right-context", &extra_right_context,
                     "Number of frames of additional right-context to add on top "
                         "of the neural net's inherent right context (may be useful in "
                         "recurrent setups");
      opts->Register("extra-left-context-initial", &extra_left_context_initial,
                     "If >= 0, overrides the --extra-left-context value at the "
                         "start of an utterance.");
      opts->Register("extra-right-context-final", &extra_right_context_final,
                     "If >= 0, overrides the --extra-right-context value at the "
                         "end of an utterance.");
      opts->Register("frame-subsampling-factor", &frame_subsampling_factor,
                     "Required if the frame-rate of the output (e.g. in 'chain' "
                         "models) is less than the frame-rate of the original "
                         "alignment.");
      opts->Register("acoustic-scale", &acoustic_scale,
                     "Scaling factor for acoustic log-likelihoods");
      opts->Register("frames-per-chunk", &frames_per_chunk,
                     "Number of frames in each chunk that is separately evaluated "
                         "by the neural net.  Measured before any subsampling, if the "
                         "--frame-subsampling-factor options is used (i.e. counts "
                         "input frames");
    }
};

class DecodableAmPosteriors: public DecodableInterface {
 public:
  /**
     This constructor takes posteriors as input.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] opts   The options class.  Warning: it includes an acoustic
                        weight, whose default is 0.1; you may sometimes want to
                        change this to 1.0.
     @param [in] trans_model  The transition model to use.  This takes care of the
                        mapping from transition-id (which is an arg to
                        LogLikelihood()) to pdf-id (which is used internally).
     @param [in] posteriors A pointer to the input posterior matrix; must be non-NULL.
  */
  DecodableAmPosteriors(const TransitionModel &trans_model,
                        const MatrixBase<BaseFloat> &posteriors);


  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual inline int32 NumFramesReady() const {
    return (*posteriors_copy_).NumRows();
  }

  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmPosteriors);
  const TransitionModel &trans_model_;
  const MatrixBase<BaseFloat> *posteriors_copy_;
};

} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_NNET3_NNET_AM_DECODABLE_POSTERIORS_H_
