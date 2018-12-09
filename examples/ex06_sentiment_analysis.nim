# Analyse movie reviews with a simple sentiment analysis via recurrent neural network

import
  ../src/arraymancer,
  random, sequtils, strformat

# Make the results reproducible by initializing a random seed
randomize(42)

# ############################################################
#
#               Loading and exploring the data
#
# ############################################################

# Load the dataset
let imdb = load_imdb(cache = true)

# The IMDB dataset contains highly polar reviews of movies.
# Either reviews consider the movie poor and rated it between 1 and 4
# or they praised it highly and rate it between 7 and 10.
echo &"Number of training samples: {imdb.train_texts.shape}"
echo &"Number of testing samples: {imdb.train_texts.shape}"

# Let's print 2 positive and 2 negative reviews
echo "First positive review: "
echo "\t", imdb.train_texts[0]
echo "Rating: ", imdb.train_labels[0]
echo "\n----\n"

echo "Second positive review: "
echo "\t", imdb.train_texts[1]
echo "Rating: ", imdb.train_labels[1]
echo "\n----\n"

echo "First negative review: "
echo "\t", imdb.train_texts[12500]
echo "Rating: ", imdb.train_labels[12500]
echo "\n----\n"

echo "Second negative review: "
echo "\t", imdb.train_texts[12501]
echo "Rating: ", imdb.train_labels[12501]
echo "\n----\n"

# ############################################################
#
#                   Preparing the data
#
# ############################################################

# Keeping the data as is will causes issue in training we need to mix
# positive and negative reviews otherwise the network will be specialized
# to the last section it was trained on.

var
  X = imdb.train_texts
  y = imdb.train_labels

block: # Shuffling
  var rng = initRand 0xDEADBEEF # We use a custom random seed for shuffling

  for i in countdown(X.shape[0], 1):
    let j = rng.rand(i) # Generate a new index between 0 and i
    swap(X[i], X[j])
    swap(y[i], y[j])

# With texts being variable length it often comes very unwieldly to deal with variable size.
# So we need to truncate test to a reasonable amount, this also ensure that if some texts
# are longer in the test set, they can still be worked on.
# Similarly, the vocabulary size must be restricted.
#
# To filter we will:
#    1. ignore stop words (i.e. words too common that don't help for sentiment analysis)
#    2. Keep

block:



# ⚠️ - The current neural network declaration syntax is too restrictive
#     for embedding layers and will be change in the future.

