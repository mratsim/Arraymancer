
import ../../src/arraymancer
import std / [unittest, strutils, times]

proc main() =
  suite "Tokenizers":
    test "Whitespace Tokenizer - Basic":
      let text = [["hello world"], ["the quick brown fox"]].toTensor()

      var tokenized = newSeq[seq[string]]()
      for word in whitespaceTokenizer(text):
          tokenized.add(word)

      check:
          tokenized[0] == @["hello", "world"]
          tokenized[1] == @["the", "quick", "brown", "fox"]

    test "Whitespace Tokenizer - Speed":
      # 1 million word text
      let sample = "something\tgreat\nsomething really terrible "
      let words = 5
      let size = 200_000
      let total =words * size
      let longText = sample.repeat(size).toTensor()

      var tokenized = newSeq[seq[string]]()
      let start = cpuTime()
      for word in whitespaceTokenizer(longText):
          tokenized.add(word)
      let stop = cpuTime()
      let wordsPerSecond = total.toFloat() / (stop - start)

      var minWps = 400_000
      when defined(release):
          miWps = 4_000_000

      check:
          len(tokenized[0]) == size * words
          # See bottom of https://nlp.stanford.edu/software/tokenizer.shtml
          # for a good comparison between stanford NLP (Java) and spaCy (cython)
          # performance on words tokenized per second
          wordsPerSecond > minWps.toFloat


main()
GC_fullCollect()
