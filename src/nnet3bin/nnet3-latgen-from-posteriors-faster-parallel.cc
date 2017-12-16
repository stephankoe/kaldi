// nnet3bin/nnet3-latgen-faster-parallel.cc

// Copyright 2012-2016   Johns Hopkins University (author: Daniel Povey)
//                2014   Guoguo Chen

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


#include "base/timer.h"
#include "base/kaldi-common.h"
#include "decoder/decoder-wrappers.h"
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-am-decodable-posteriors.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"



int main(int argc, char *argv[]) {
  // note: making this program work with GPUs is as simple as initializing the
  // device, but it probably won't make a huge difference in speed for typical
  // setups.
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using posteriors.\n"
        "Usage: nnet3-latgen-from-posteriors-faster-parallel [options] <model-in> <fst-in|fsts-rspecifier> <posteriors-rspecifier>"
        " <lattice-wspecifier> [ <words-wspecifier> [<alignments-wspecifier>] ]\n";
    ParseOptions po(usage);

    Timer timer;
    bool allow_partial = false;
    TaskSequencerConfig sequencer_config; // has --num-threads option
    LatticeFasterDecoderConfig config;
    NnetPosteriorOptions decodable_opts;

    std::string word_syms_filename;
    std::string utt2spk_rspecifier;
    sequencer_config.Register(&po);
    config.Register(&po);
    decodable_opts.Register(&po);
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        lattice_wspecifier = po.GetArg(4),
        words_wspecifier = po.GetOptArg(5),
        alignment_wspecifier = po.GetOptArg(6);

    TaskSequencer<DecodeUtteranceLatticeFasterClass> sequencer(sequencer_config);
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
    }

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader posteriors_reader(posteriors_rspecifier);

      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
      timer.Reset();

      {
        LatticeFasterDecoder decoder(*decode_fst, config);

        for (; !posteriors_reader.Done(); posteriors_reader.Next()) {
          std::string utt = posteriors_reader.Key();
          const Matrix<BaseFloat> &posteriors (posteriors_reader.Value());
          if (posteriors.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }

          LatticeFasterDecoder *decoder =
              new LatticeFasterDecoder(*decode_fst, config);

          DecodableInterface *nnet_decodable = new
              DecodableAmPosteriors(trans_model, posteriors);

          DecodeUtteranceLatticeFasterClass *task =
              new DecodeUtteranceLatticeFasterClass(
                  decoder, nnet_decodable, // takes ownership of these two.
                  trans_model, word_syms, utt, decodable_opts.acoustic_scale,
                  determinize, allow_partial, &alignment_writer, &words_writer,
                   &compact_lattice_writer, &lattice_writer,
                   &tot_like, &frame_count, &num_success, &num_fail, NULL);

          sequencer.Run(task); // takes ownership of "task",
                               // and will delete it when done.
        }
      }
      sequencer.Wait(); // Waits for all tasks to be done.
      delete decode_fst;
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader posteriors_reader(posteriors_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!posteriors_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no features available.";
          num_fail++;
          continue;
        }
        const Matrix<BaseFloat> &posteriors = posteriors_reader.Value(utt);
        if (posteriors.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          continue;
        }

        // the following constructor takes ownership of the FST pointer so that
        // it is deleted when 'decoder' is deleted.
        LatticeFasterDecoder *decoder =
            new LatticeFasterDecoder(config, fst_reader.Value().Copy());

        DecodableInterface *nnet_decodable = new
            DecodableAmPosteriors(trans_model, posteriors);

        DecodeUtteranceLatticeFasterClass *task =
            new DecodeUtteranceLatticeFasterClass(
                decoder, nnet_decodable, // takes ownership of these two.
                trans_model, word_syms, utt, decodable_opts.acoustic_scale,
                determinize, allow_partial, &alignment_writer, &words_writer,
                &compact_lattice_writer, &lattice_writer,
                &tot_like, &frame_count, &num_success, &num_fail, NULL);

        sequencer.Run(task); // takes ownership of "task",
        // and will delete it when done.
      }
      sequencer.Wait(); // Waits for all tasks to be done.
    }

    kaldi::int64 input_frame_count =
        frame_count * decodable_opts.frame_subsampling_factor;

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken " << elapsed
              << "s: real-time factor assuming 100 feature frames/sec is "
              << (sequencer_config.num_threads * elapsed * 100.0 /
                  input_frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over "
              << frame_count << " frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
