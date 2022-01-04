import ../src/arraymancer, random

# This is an early minimum viable example of handwritten digits recognition.
# It uses convolutional neural networks to achieve high accuracy.
#
# Data files (MNIST) can be downloaded here http://yann.lecun.com/exdb/mnist/
# and must be decompressed in "./build/" (or change the path "build/..." below)
#

# Make the results reproducible by initializing a random seed
randomize(42)

let
  ctx = newContext Tensor[float32] # Autograd/neural network graph
  n = 32                           # Batch size

let
  mnist = load_mnist(cache = true)
  # Training data is 60k 28x28 greyscale images from 0-255,
  # neural net prefers input rescaled to [0, 1] or [-1, 1]
  x_train = mnist.train_images.astype(float32) / 255'f32

  # Change shape from [N, H, W] to [N, C, H, W], with C = 1 (unsqueeze). Convolution expect 4d tensors
  # And store in the context to track operations applied and build a NN graph
  X_train = ctx.variable x_train.unsqueeze(1)

  # Labels are uint8, we must convert them to int
  y_train = mnist.train_labels.astype(int)

  # Idem for testing data (10000 images)
  x_test = mnist.test_images.astype(float32) / 255'f32
  X_test = ctx.variable x_test.unsqueeze(1)
  y_test = mnist.test_labels.astype(int)

# Configuration of the neural network
network DemoNet:
  layers:
    cv1:        Conv2D(@[1, 28, 28], 20, (5, 5))
    mp1:        Maxpool2D(cv1.out_shape, (2,2), (0,0), (2,2))
    cv2:        Conv2D(mp1.out_shape, 50, (5, 5))
    mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
    fl:         Flatten(mp2.out_shape)
    hidden:     Linear(fl.out_shape[0], 500)
    classifier: Linear(500, 10)
  forward x:
    x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.classifier

let model = ctx.init(DemoNet)

# Stochastic Gradient Descent (API will change)
let optim = model.optimizerSGD(learning_rate = 0.01'f32)

# Learning loop
for epoch in 0 ..< 5:
  for batch_id in 0 ..< X_train.value.shape[0] div n: # some at the end may be missing, oh well ...
    # minibatch offset in the Tensor
    let offset = batch_id * n
    let x = X_train[offset ..< offset + n, _]
    let target = y_train[offset ..< offset + n]

    # Running through the network and computing loss
    let clf = model.forward(x)
    let loss = clf.sparse_softmax_cross_entropy(target)

    if batch_id mod 200 == 0:
      # Print status every 200 batches
      echo "Epoch is: " & $epoch
      echo "Batch id: " & $batch_id
      echo "Loss is:  " & $loss.value[0]

    # Compute the gradient (i.e. contribution of each parameter to the loss)
    loss.backprop()

    # Correct the weights now that we have the gradient information
    optim.update()

  # Validation (checking the accuracy/generalization of our model on unseen data)
  ctx.no_grad_mode:
    echo "\nEpoch #" & $epoch & " done. Testing accuracy"

    # To avoid using too much memory we will compute accuracy in 10 batches of 1000 images
    # instead of loading 10 000 images at once
    var score = 0.0
    var loss = 0.0
    for i in 0 ..< 10:
      let y_pred = model.forward(X_test[i*1000 ..< (i+1)*1000, _]).value.softmax.argmax(axis = 1).squeeze
      score += y_pred.accuracy_score(y_test[i*1000 ..< (i+1)*1000])

      loss += model.forward(X_test[i*1000 ..< (i+1)*1000, _]).sparse_softmax_cross_entropy(y_test[i*1000 ..< (i+1)*1000]).value.unsafe_raw_offset[0]
    score /= 10
    loss /= 10
    echo "Accuracy: " & $(score * 100) & "%"
    echo "Loss:     " & $loss
    echo "\n"


############# Output ############

# Epoch is: 0
# Batch id: 0
# Loss is:  2.613911151885986
# Epoch is: 0
# Batch id: 200
# Loss is:  0.310189962387085
# Epoch is: 0
# Batch id: 400
# Loss is:  0.1142476052045822
# Epoch is: 0
# Batch id: 600
# Loss is:  0.1852415204048157
# Epoch is: 0
# Batch id: 800
# Loss is:  0.1070618182420731
# Epoch is: 0
# Batch id: 1000
# Loss is:  0.2025316953659058
# Epoch is: 0
# Batch id: 1200
# Loss is:  0.1810623109340668
# Epoch is: 0
# Batch id: 1400
# Loss is:  0.09978601336479187
# Epoch is: 0
# Batch id: 1600
# Loss is:  0.09480990469455719
# Epoch is: 0
# Batch id: 1800
# Loss is:  0.2068064212799072
# 
# Epoch #0 done. Testing accuracy
# Accuracy: 96.82000000000001%
# Loss:     0.09608472418040037
# 
# 
# Epoch is: 1
# Batch id: 0
# Loss is:  0.03820636868476868
# Epoch is: 1
# Batch id: 200
# Loss is:  0.05903942883014679
# Epoch is: 1
# Batch id: 400
# Loss is:  0.06314512342214584
# Epoch is: 1
# Batch id: 600
# Loss is:  0.07504448294639587
# Epoch is: 1
# Batch id: 800
# Loss is:  0.03850477933883667
# Epoch is: 1
# Batch id: 1000
# Loss is:  0.1440025717020035
# Epoch is: 1
# Batch id: 1200
# Loss is:  0.2018975913524628
# Epoch is: 1
# Batch id: 1400
# Loss is:  0.04561902582645416
# Epoch is: 1
# Batch id: 1600
# Loss is:  0.07066516578197479
# Epoch is: 1
# Batch id: 1800
# Loss is:  0.1373588591814041
# 
# Epoch #1 done. Testing accuracy
# Accuracy: 97.74000000000001%
# Loss:     0.06817570002749562
# 
# 
# Epoch is: 2
# Batch id: 0
# Loss is:  0.01814174652099609
# Epoch is: 2
# Batch id: 200
# Loss is:  0.03460906445980072
# Epoch is: 2
# Batch id: 400
# Loss is:  0.05443669855594635
# Epoch is: 2
# Batch id: 600
# Loss is:  0.04482628405094147
# Epoch is: 2
# Batch id: 800
# Loss is:  0.02421820163726807
# Epoch is: 2
# Batch id: 1000
# Loss is:  0.1148378998041153
# Epoch is: 2
# Batch id: 1200
# Loss is:  0.2140489369630814
# Epoch is: 2
# Batch id: 1400
# Loss is:  0.02446934580802917
# Epoch is: 2
# Batch id: 1600
# Loss is:  0.05318602919578552
# Epoch is: 2
# Batch id: 1800
# Loss is:  0.1024059653282166
# 
# Epoch #2 done. Testing accuracy
# Accuracy: 98.10000000000001%
# Loss:     0.05548303546383977
# 
# 
# Epoch is: 3
# Batch id: 0
# Loss is:  0.01244649291038513
# Epoch is: 3
# Batch id: 200
# Loss is:  0.02020946145057678
# Epoch is: 3
# Batch id: 400
# Loss is:  0.04690191149711609
# Epoch is: 3
# Batch id: 600
# Loss is:  0.03282114863395691
# Epoch is: 3
# Batch id: 800
# Loss is:  0.01641204953193665
# Epoch is: 3
# Batch id: 1000
# Loss is:  0.09525816142559052
# Epoch is: 3
# Batch id: 1200
# Loss is:  0.2170524597167969
# Epoch is: 3
# Batch id: 1400
# Loss is:  0.0137055516242981
# Epoch is: 3
# Batch id: 1600
# Loss is:  0.04224002361297607
# Epoch is: 3
# Batch id: 1800
# Loss is:  0.07904504239559174
# 
# Epoch #3 done. Testing accuracy
# Accuracy: 98.38%
# Loss:     0.04833780862390995
# 
# 
# Epoch is: 4
# Batch id: 0
# Loss is:  0.009280085563659668
# Epoch is: 4
# Batch id: 200
# Loss is:  0.01469701528549194
# Epoch is: 4
# Batch id: 400
# Loss is:  0.04068432748317719
# Epoch is: 4
# Batch id: 600
# Loss is:  0.02430866658687592
# Epoch is: 4
# Batch id: 800
# Loss is:  0.01234877109527588
# Epoch is: 4
# Batch id: 1000
# Loss is:  0.07774896919727325
# Epoch is: 4
# Batch id: 1200
# Loss is:  0.2160461992025375
# Epoch is: 4
# Batch id: 1400
# Loss is:  0.008712261915206909
# Epoch is: 4
# Batch id: 1600
# Loss is:  0.03578883409500122
# Epoch is: 4
# Batch id: 1800
# Loss is:  0.06306551396846771
# 
# Epoch #4 done. Testing accuracy
# Accuracy: 98.48999999999999%
# Loss:     0.04407740226015448
# 
# 
# 
# ________________________________________________________
# Executed in   36.46 mins    fish           external
#    usr time   36.40 mins  344.00 micros   36.40 mins
#    sys time    0.01 mins  192.00 micros    0.01 mins
