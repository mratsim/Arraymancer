
===================================
Working with OpenCL and Cuda in Nim
===================================

*Date: May 6, 2018, by Mamy André-Ratsimbazafy*

Arraymancer is a tensor library I’m writing from the ground up in Nim.
Cuda support was added in v0.3 last December, I just released the new v0.4 with OpenCL support.

I’d like to share a bit of my experience on working in OpenCL through Nim.
First of all, you have to know that none of the big guys (Google Tensorflow, Facebook PyTorch,
Apache/Amazon MxNet, Microsoft CNTK or even Intel/AMD) has first class OpenCL support.

Why? Probably because Nvidia is providing superb tools and documentation for frameworks developers.
Also Cuda can leerage a few C++ facilities like generics and function objects that I use heavily for generic code.

For example in Nim+Cuda I define element-wise functions like the following and
pass it to a higher-order function that will apply it element-wise on 3 tensors:

.. code:: nim
    # Binary op
    # Does C[i] = A[i] `op` B[i]
    template cuda_binary_op(op_name, op_symbol: string)=
      {.emit:["""
      template<typename T>
      struct """,op_name,"""{
      __device__ __forceinline__ void operator()(
          T *  __restrict__ dst,
          const T *  __restrict__ A,
          const T *  __restrict__ B){
          *dst = __ldg(A)""", op_symbol, """ __ldg(B);
          }
      };
      """].}

You can see here the advantage of C++: ``typename T`` to template over int/float/double and
higher-order functions/function object for cleaner code.
You can also see that Nim can directly inline C++ code with ``emit`` and I even templatize the operation_name.

Now what about OpenCL? Unfortunately C doesn’t offer something similar and requires a lot of boilerplate.
The alternative, the C++ official OpenCL API and implementation: SYCL is
very experimental and I am not sure how it works on actual GPUs.

However thanks to Nim metaprogramming, squashing the C boilerplate is super easy. Here is an example kernel to do C = A op B

.. code:: nim
    template gen_cl_apply3*(kern_name, ctype, op: string): string =
      ## Generates an OpenCL kernel for an elementwise binary infix operations (like +, -, ...)
      ## Input:
      ##   - The C type
      ##   - The C kernel name (this only helps debugging the C code)
      ##   - The C operation (+, -, ...)


      opencl_getIndexOfElementID() & """
      __kernel
      void """ & kern_name &
              """(const int rank,
                  const int len,
                  __global const int * restrict dst_shape,
                  __global const int * restrict dst_strides,
                  const int dst_offset,
                  __global       """ & ctype & """ * restrict const dst_data,
                  __global const int * restrict A_shape,
                  __global const int * restrict A_strides,
                  const int A_offset,
                  __global const """ & ctype & """ * restrict const A_data,
                  __global const int * restrict B_shape,
                  __global const int * restrict B_strides,
                  const int B_offset,
                  __global const """ & ctype & """ * restrict const B_data)
      {
        // Grid-stride loop
        for (int elemID = get_global_id(0);
        elemID < len;
        elemID += get_global_size(0)) {
          const int dst_real_idx = opencl_getIndexOfElementID(rank, dst_shape, dst_strides, dst_offset, elemID);
          const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
          const int B_real_idx = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);

          dst_data[dst_real_idx] = A_data[A_real_idx] """ & op & """ B_data[B_real_idx];
        }
      }
      """

And write a few generic lines of code to deal with the data on the device
(especially ``opencl_getIndexOfElementID`` which convert ``foo[1, 2, 3]`` into ``foo.data[456]``
depending on the tensor shape.

Afterwards, all my operations are easily added in one line:

* kind of function (infix: C = A op B or in-place A += B or A \*= B)
* Nim type
* C type
* Nim operator (for operator overloading)
* OpenCL kernel name
* OpenCL operation

.. code:: nim
    genClInfixOp(float32, "float", `+`, "clAdd", "+")
    genClInfixOp(float64, "double", `+`, "clAdd", "+")
    genClInfixOp(float32, "float", `-`, "clSub", "-")
    genClInfixOp(float64, "double", `-`, "clSub", "-")

    genClInPlaceOp(float32, "float", `+=`, "clAdd", "+=")
    genClInPlaceOp(float64, "double", `+=`, "clAdd", "+=")
    genClInPlaceOp(float32, "float", `-=`, "clSub", "-=")
    genClInPlaceOp(float64, "double", `-=`, "clSub", "-=")

Next steps? Create unary operation higher-order functions and add cos/sin/ln/exp in just 2 lines of code each.
Furthermore allow lifting any unary operation to operations on whole tensors with a map function,
expose it so that OpenCL tensors are easily customizable.

After using Nim + OpenCL, I actually realized that using C++ function objects was overengineering.

To conclude, at the moment, I am convinced that the best language to work with GPUs is Nim.

Oh, and for those who wants to see real Nim code for neural networks, here is a Fizzbuzz in Nim using neural networks (I didn’t implement it on GPU yet though)

.. code:: nim
    # A port to Arraymancer of Joel Grus hilarious FizzBuzz in Tensorflow:
    # http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

    # Interviewer: Welcome, can I get you a coffee or anything? Do you need a break?
    # ...
    # Interviewer: OK, so I need you to print the numbers from 1 to 100,
    #              except that if the number is divisible by 3 print "fizz",
    #              if it's divisible by 5 print "buzz", and if it's divisible by 15 print "fizzbuzz".

    # Let's start with standard imports
    import ../src/arraymancer, math, strformat

    # We want to input a number and output the correct "fizzbuzz" representation
    # ideally the input is a represented by a vector of real values between 0 and 1
    # One way to do that is by using the binary representation of number
    func binary_encode(i: int, num_digits: int): Tensor[float32] =
      result = newTensor[float32](1, num_digits)
      for d in 0 ..< num_digits:
        result[0, d] = float32(i shr d and 1)

    # For the input, we distinguishes 4 cases, nothing, fizz, buzz and fizzbuzz.
    func fizz_buzz_encode(i: int): int =
      if   i mod 15 == 0: return 3 # fizzbuzz
      elif i mod  5 == 0: return 2 # buzz
      elif i mod  3 == 0: return 1 # fizz
      else              : return 0

    # Next, let's generate training data, we don't want to train on 1..100, that's our test values
    # We can't tell the neural net the truth values it must discover the logic by itself.
    # so we use values between 101 and 1024 (2^10)
    const NumDigits = 10

    var x_train = newTensor[float32](2^NumDigits - 101, NumDigits)
    var y_train = newTensor[int](2^NumDigits - 101)

    for i in 101 ..< 2^NumDigits:
      x_train[i - 101, _] = binary_encode(i, NumDigits)
      y_train[i - 101] = fizz_buzz_encode(i)

    # How many neurons do we need to change a light bulb, sorry do a division? let's pick ...
    const NumHidden = 100

    # Let's setup our neural network context, variables and model
    let
      ctx = newContext Tensor[float32]
      X   = ctx.variable x_train

    network ctx, FizzBuzzNet:
      layers:
        hidden: Linear(NumDigits, NumHidden)
        output: Linear(NumHidden, 4)
      forward x:
        x.hidden.relu.output

    let model = ctx.init(FizzBuzzNet)
    let optim = model.optimizerSGD(0.05'f32)

    func fizz_buzz(i: int, prediction: int): string =
      [$i, "fizz", "buzz", "fizzbuzz"][prediction]

    # Phew, finally ready to train, let's pick the batch size and number of epochs
    const BatchSize = 128
    const Epochs    = 2500

    # And let's start training the network
    for epoch in 0 ..< Epochs:
      # Here I should probably shuffle the input data.
      for start_batch in countup(0, x_train.shape[0]-1, BatchSize):

        # Pick the minibatch
        let end_batch = min(x_train.shape[0]-1, start_batch + BatchSize)
        let X_batch = X[start_batch ..< end_batch, _]
        let target = y_train[start_batch ..< end_batch]

        # Go through the model
        let clf = model.forward(X_batch)

        # Go through our cost function
        let loss = clf.sparse_softmax_cross_entropy(target)

        # Backpropagate the errors and let the optimizer fix them.
        loss.backprop()
        optim.update()

      # Let's see how we fare:
      ctx.no_grad_mode:
        echo &"\nEpoch #{epoch} done. Testing accuracy"

        let y_pred = model
                      .forward(X)
                      .value
                      .softmax
                      .argmax(axis = 1)
                      .squeeze

        let score = y_pred.accuracy_score(y_train)
        echo &"Accuracy: {score:.3f}%"
        echo "\n"


    # Our network is trained, let's see if it's well behaved

    # Now let's use what we really want to fizzbuzz, numbers from 1 to 100
    var x_buzz = newTensor[float32](100, NumDigits)
    for i in 1 .. 100:
      x_buzz[i - 1, _] = binary_encode(i, NumDigits)

    # Wrap them for neural net
    let X_buzz = ctx.variable x_buzz

    # Pass it through the network
    ctx.no_grad_mode:
      let y_buzz = model
                    .forward(X_buzz)
                    .value
                    .softmax
                    .argmax(axis = 1)
                    .squeeze

    # Extract the answer
    var answer: seq[string] = @[]

    for i in 1..100:
      answer.add fizz_buzz(i, y_buzz[i - 1])

    echo answer
    # @["1", "2", "fizz", "4", "buzz", "6", "7", "8", "fizz", "10",
    #   "11", "12", "13", "14", "15", "16", "17", "fizz", "19", "buzz",
    #   "fizz", "22", "23", "24", "buzz", "26", "fizz", "28", "29", "30",
    #   "31", "32", "fizz", "34", "buzz", "36", "37", "38", "39", "40",
    #   "41", "fizz", "43", "44", "fizzbuzz", "46", "47", "fizz", "49", "50",
    #   "fizz", "52","53", "54", "buzz", "56", "fizz", "58", "59", "fizzbuzz",
    #   "61", "62", "63", "64", "buzz", "fizz", "67", "68", "fizz", "buzz",
    #   "71", "fizz", "73", "74", "75", "76", "77","fizz", "79", "buzz",
    #   "fizz", "82", "83", "fizz", "buzz", "86", "fizz", "88", "89", "90",
    #   "91", "92", "fizz", "94", "buzz", "fizz", "97", "98", "fizz", "buzz"]

    # I guess 100 neurons are not enough to learn multiplication :/.

Thank you for your attention and your support,

Be sure to try `Nim<https://nim-lang.org/>`__ and `Arraymancer<https://github.com/mratsim/Arraymancer>`__!
