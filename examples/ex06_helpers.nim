# Analyse movie reviews with a simple sentiment analysis via recurrent neural network
# Helper functions

# Those functions will ideally be transformed into robust primitives
# once their API is ironed out

import
  strutils, tables, math,
  ../src/arraymancer

# ############################################################
#
#                      English stop_words
#
# ############################################################

# Note, to preserve context we will not use stop words.
# Some words like "not" must be preserved for sentiment analysis.
# and others might actually give sentiment information like "again" or "behind"
# See also: https://www.cs.cmu.edu/~ark/EMNLP-2015/proceedings/WASSA/pdf/WASSA14.pdf
#
# Instead we will rely on stochastic metadata like term frequencies
# and inverse document frequencies to keep the words that hold the most meaning
# in case we go over our vocabulary size limit: https://cran.r-project.org/web/packages/tidytext/vignettes/tf_idf.html

# Source - spaCy - https://github.com/explosion/spaCy/blob/e5685d98a2644e515df9e21bc1aa8f003d5a02c4/spacy/lang/en/stop_words.py

# import sets
# const StopWords = """
# a about above across after afterwards again against all almost alone along
# already also although always am among amongst amount an and another any anyhow
# anyone anything anyway anywhere are around as at
# back be became because become becomes becoming been before beforehand behind
# being below beside besides between beyond both bottom but by
# call can cannot ca could
# did do does doing done down due during
# each eight either eleven else elsewhere empty enough even ever every
# everyone everything everywhere except
# few fifteen fifty first five for former formerly forty four from front full
# further
# get give go
# had has have he hence her here hereafter hereby herein hereupon hers herself
# him himself his how however hundred
# i if in indeed into is it its itself
# keep
# last latter latterly least less
# just
# made make many may me meanwhile might mine more moreover most mostly move much
# must my myself
# name namely neither never nevertheless next nine no nobody none noone nor not
# nothing now nowhere
# of off often on once one only onto or other others otherwise our ours ourselves
# out over own
# part per perhaps please put
# quite
# rather re really regarding
# same say see seem seemed seeming seems serious several she should show side
# since six sixty so some somehow someone something sometime sometimes somewhere
# still such
# take ten than that the their them themselves then thence there thereafter
# thereby therefore therein thereupon these they third this those though three
# through throughout thru thus to together too top toward towards twelve twenty
# two
# under until up unless upon us used using
# various very very via was we well were what whatever when whence whenever where
# whereafter whereas whereby wherein whereupon wherever whether which while
# whither who whoever whole whom whose why will with within without would
# yet you your yours yourself yourselves
# """.split(NewLines + Whitespace).toSet

# ############################################################
#
#                  Vocabulary statistics
#
# ############################################################

type VocabStats = object
  corpus_counts: CountTable[string]         # Count how often a word appear in the whole dataset
  unique_corpus_counts: CountTable[string]  # Count in how many different corpus a word appear
  doc_counts: seq[CountTable[string]]       # Count how often a word appear per document of the dataset
  word_counts: seq[int]                     # Count the number of words per doc

  # term_freqs: seq[Table[string, float32]]   # Frequency of each term with regard to their per document appearance
  # inverse_doc_freqs: Table[string, float32] # ln(number of docs / (1 + docs with the terms)) - We add one to prevent a division by 0
  # tf_idf: seq[Table[string, float32]]       # = term_freqs * inverse_doc_freqs i.e. how much info this term bring

  max_tf_idf: OrderedTable[string, float32] # We keep track of the maximum tf_idf a term ever has in the corpus

# Some optimisations to avoid Nim grows table capacity too often.
const
  # Apparently we had 171 476 words in the Oxford English Dictionarry from 1989.
  # We can reasonably expect that much less are in use.
  #
  # Caveats: words like "awesome", "awesome!" and "awesome!!" are considered different
  #          if we use a simple whitespace word tokenizer.
  #
  # Also Nim's tables size must be power of 2.
  # Let's use 2^17 = 131 072. With ints taking 8 bytes each, this represent a reasonable 1 MB of memory
  IniSizeCorpus = 1 shl 17

  # For individual reviews, we can expect that people use at most 20 min to type them.
  # The average typing speed is 40 words per minute --> 800 words.
  # Using the 80/20 Pareto heuristic, let's assume that 20% of those 800 are unique,
  # and the rest are reused multiple times. This corresponds to 160 words.
  # A size of 256 would only leave a 76 words buffer for repeated words so let's use 512
  IniSizeReview = 512

  # Replace some strings with either nothing or a splitting character
  # Note: this is simple but very slow as we need loop twice over each input.
  # A specialized parser+splitter would be much faster
  sanitizeWords = {
      "<br />": "\n",
      "<br/>": "\n",
    }

func corpusStats(corpus: Tensor[string]): VocabStats =
  ## TODO - too slow

  # 1. Get the count of each word
  result.corpus_counts = initCountTable[string](IniSizeCorpus)
  result.unique_corpus_counts = initCountTable[string](IniSizeCorpus)

  for review in corpus:
    result.doc_counts.add initCountTable[string](IniSizeReview)

    var word_count = 0
    for word in split(review.multiReplace(sanitizeWords)):
      result.corpus_counts.inc word
      result.doc_counts[^1].inc word
      inc word_count

    result.word_counts.add word_count

    # Update the unique corpus count with each word that appeared at least once.
    for word in result.doc_counts[^1].keys:
      result.unique_corpus_counts.inc word

  # 2. Compute the max tf_idf each word ever had from all codument in the corpus
  result.max_tf_idf = initOrderedTable[string, float32](initialSize = result.corpus_counts.len.rightSize)

  for doc_id, revStats in result.doc_counts:
    for word, count in revStats.pairs:
      let tf = count.float32 / result.word_counts[doc_id].float32
      let idf = ln(result.doc_counts.len.float32 / float32(1 + result.unique_corpus_counts[word]))

      let current_tfidf = result.max_tf_idf.getOrDefault(word, 0)
      let tfidf = tf * idf
      if tfidf > current_tfidf:
        result.max_tf_idf[word] = tfidf

when isMainModule:
  let imdb = load_imdb()

  var vocab = imdb.train_texts.corpusStats()

  vocab.corpus_counts.sort()
  vocab.unique_corpus_counts.sort()
  vocab.max_tf_idf.sort(system.cmp) # Note we should sort in reverse

  import strformat

  block:
    echo "\n### Most frequent words in all documents:"
    var i = 0
    for k, v in vocab.corpus_counts.pairs:
      echo &"{k:>15}    {v:>6}"
      if i >= 20:
        break
      else:
        inc i

  block:
    echo "\n### Most common words in all documents (counted once per doc):"
    var i = 0
    for k, v in vocab.unique_corpus_counts.pairs:
      echo &"{k:>15}    {v:>6}"
      if i >= 20:
        break
      else:
        inc i

  block:
    echo "\n### Most relevant words in all documents (counted once per doc):"
    var i = 0
    for k, v in vocab.max_tf_idf.pairs:
      echo &"{k:>15}    {v:>6.3f}"
      if i >= 20:
        break
      else:
        inc i


  ## This is much too slow
  ## Note that tfidf output is wrong because I forgot to reverse the sorting

  # ### Most frequent words in all documents:
  #             the    287174
  #               a    155122
  #             and    152719
  #              of    142986
  #              to    132583
  #              is    103248
  #              in     85593
  #               I     69631
  #            that     64568
  #                     57479
  #            this     57245
  #              it     54454
  #             was     46707
  #              as     42521
  #            with     41728
  #             for     41078
  #             The     41007
  #             but     33799
  #              on     30771
  #           movie     30516
  #             are     28503

  # ### Most common words in all documents (counted once per doc):
  #             the     24669
  #               a     24048
  #             and     23967
  #              of     23675
  #              to     23406
  #              is     22262
  #              in     21648
  #            this     20700
  #            that     19542
  #              it     19043
  #               I     18324
  #             for     17354
  #            with     17064
  #             The     17026
  #             but     16170
  #             was     15994
  #              as     15379
  #              on     14948
  #                     14665
  #            have     14014
  #              be     13716

  # ### Most relevant words in all documents (counted once per doc):
  #                     0.094
  #       A     0.014
  #            own     0.018
  #               !     4.882
  #              !!     0.135
  #             !!!     0.234
  #            !!!!     0.172
  #           !!!!!     0.062
  #          !!!!!!     0.210
  #         !!!!!!!     0.278
  #        !!!!!!!!     0.149
  #       !!!!!!!!!     0.298
  #      !!!!!!!!!!     0.103
  #     !!!!!!!!!!!     0.095
  #   !!!!!!!!!!!!!     0.081
  #  !!!!!!!!!!!!!!     0.103
  # !!!!!!!!!!!!!!!!!!!!!!     0.103
  # !!!!!!!!!!!!!!!!!!!!!!!!!!     0.131
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     0.103
  #           !!!!"     0.066
  #            !!!)     0.026
