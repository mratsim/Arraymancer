import
  std / [os, random, times, strformat, algorithm, tables, math, parseopt, strutils],
  ../src/arraymancer

# transformer

type
  TransformerBlock[T] = object
    ln1*: LayerNorm[T]
    attn*: MultiHeadAttention[T]
    ln2*: LayerNorm[T]
    mlp_fc1*: Linear[T]
    mlp_fc2*: Linear[T]

  TransformerLM[T] = object
    tok_emb*: Embedding[T]
    pos_emb*: Embedding[T]
    blocks*: seq[TransformerBlock[T]]
    ln_f*: LayerNorm[T]
    head*: Linear[T]
    max_seq_len*: int

proc initBlock[T](ctx: Context[Tensor[T]], dim, heads: int, mult: int = 4): TransformerBlock[T] =
  result.ln1 = ctx.init(LayerNorm[T], dim)
  result.attn = ctx.init(MultiHeadAttention[T], dim, heads)
  result.ln2 = ctx.init(LayerNorm[T], dim)
  result.mlp_fc1 = ctx.init(Linear[T], dim, dim * mult)
  result.mlp_fc2 = ctx.init(Linear[T], dim * mult, dim)

proc forward[T](self: TransformerBlock[T], x: Variable[Tensor[T]]): Variable[Tensor[T]] =
  # pre-ln attention
  let x1 = x + self.attn.forward(self.ln1.forward(x), is_causal = true)

  # pre-ln ffn
  let h = self.mlp_fc1.forward(self.ln2.forward(x1)).relu()
  result = x1 + self.mlp_fc2.forward(h)

proc initTransformerLM[T](
  ctx: Context[Tensor[T]],
  vocab_size, dim, max_seq_len, layers, heads: int
): TransformerLM[T] =
  result.tok_emb = ctx.init(Embedding[T], vocab_size, dim)
  result.pos_emb = ctx.init(Embedding[T], max_seq_len, dim)
  result.blocks = newSeq[TransformerBlock[T]](layers)
  for i in 0 ..< layers:
    result.blocks[i] = ctx.initBlock(dim, heads)
  result.ln_f = ctx.init(LayerNorm[T], dim)
  result.head = ctx.init(Linear[T], dim, vocab_size)
  result.max_seq_len = max_seq_len

proc forward[T](self: TransformerLM[T], tokens: Tensor[int]): Variable[Tensor[T]] =
  let n = tokens.shape[1]

  # token & position embeddings
  var pos = newTensor[int]([1, n])
  for i in 0 ..< n: pos[0, i] = i

  var x = self.tok_emb.forward(tokens) +. self.pos_emb.forward(pos)

  # transformer blocks
  for blk in self.blocks:
    x = blk.forward(x)

  # head
  result = self.head.forward(self.ln_f.forward(x))

# sampling

proc sample[T: SomeFloat](probs: Tensor[T], rng: var Rand): int =
  let u = T(rng.rand(1.0))
  var c = 0.T
  for i in 0 ..< probs.size:
    c += probs[i]
    if u <= c: return i
  return probs.size - 1

proc generate[T](
  ctx: Context[Tensor[T]],
  model: TransformerLM[T],
  prompt: string,
  charToIx: Table[char, int],
  ixToChar: seq[char],
  length: int = 250,
  temperature: T = 0.7.T
): string =
  doAssert prompt.len > 0, "prompt must not be empty"
  var rng = initRand(42)
  var tokens = newSeq[int]()
  for ch in prompt:
    tokens.add(if ch in charToIx: charToIx[ch] else: 0)

  result = ""

  ctx.no_grad_mode:
    for _ in 0 ..< length:
      let curr = min(tokens.len, model.max_seq_len)
      let offset = tokens.len - curr
      var inp = newTensor[int]([1, curr])
      for i in 0 ..< curr:
        inp[0, i] = tokens[offset + i]

      # logits
      let logits = model.forward(inp)
      let v = logits.value.shape[^1]

      # sample last token
      var last = newTensor[T]([v])
      for i in 0 ..< v:
        last[i] = logits.value[0, curr - 1, i] / temperature

      let next_id = sample(last.softmax(), rng)
      tokens.add next_id
      result.add ixToChar[next_id]

# main

proc main() =
  var
    steps = 2000
    length = 350
    temperature = 0.65'f32
    prompt = "ROMEO:\n"
    sampleEvery = 0

  for kind, key, val in getopt():
    case kind
    of cmdLongOption, cmdShortOption:
      case key
      of "steps": steps = parseInt(val)
      of "length": length = parseInt(val)
      of "temperature": temperature = parseFloat(val).float32
      of "prompt": prompt = val
      of "sample-every": sampleEvery = parseInt(val)
      else: discard
    else: discard

  if sampleEvery <= 0:
    sampleEvery = max(1, steps div 4)

  let path = currentSourcePath().parentDir / "ex06_shakespeare_input.txt"
  if not fileExists(path):
    echo "Missing: ", path
    quit(1)

  let text = readFile(path)

  # vocab
  var chars: seq[char] = @[]
  for ch in text:
    if ch notin chars: chars.add ch
  chars.sort()

  let vocab_size = chars.len
  var charToIx = initTable[char, int]()
  var ixToChar = newSeq[char](vocab_size)
  for i, ch in chars:
    charToIx[ch] = i
    ixToChar[i] = ch

  var data = newTensor[int]([text.len])
  for i, ch in text:
    data[i] = charToIx[ch]

  # parameters
  const
    dim = 128
    heads = 4
    layers = 3
    seq_len = 64
    batch_size = 32
    lr = 0.0015'f32

  echo &"Shakespeare ({text.len} chars, vocab {vocab_size}) | Transformer: dim={dim}, heads={heads}, layers={layers}, ctx={seq_len}"

  let ctx = newContext Tensor[float32]
  let model = ctx.initTransformerLM(vocab_size, dim, seq_len, layers, heads)
  var optim = model.optimizer(Adam, learning_rate = lr)

  var rng = initRand(1337)
  let maxStart = data.len - seq_len - 1

  # train
  let t0 = epochTime()

  for step in 1 .. steps:
    var x = newTensor[int]([batch_size, seq_len])
    var y = newTensor[int]([batch_size, seq_len])

    for b in 0 ..< batch_size:
      let start = rng.rand(0 ..< maxStart)
      for s in 0 ..< seq_len:
        x[b, s] = data[start + s]
        y[b, s] = data[start + s + 1]

    # loss & backward
    let logits = model.forward(x).reshape(batch_size * seq_len, vocab_size)
    let loss = logits.sparse_softmax_cross_entropy(y.reshape(batch_size * seq_len))
    let lossVal = loss.value[0]

    loss.backprop()
    optim.update()

    if step == 1 or step mod 100 == 0 or step == steps:
      echo &"step {step:3d}/{steps} | loss {lossVal:.4f} | {epochTime() - t0:.1f}s"

    if step mod sampleEvery == 0 or step == steps:
      echo &"--- sample @ step {step} ---"
      echo &"prompt: {prompt.escape}"
      echo "--- generated ---"
      echo ctx.generate(model, prompt, charToIx, ixToChar, length = min(150, length), temperature = temperature)

  echo &"\ntrained in {epochTime() - t0:.1f}s\n"

  # generate
  echo "--- Generated Shakespeare ---"
  echo &"prompt: {prompt.escape}"
  echo "--- generated ---"
  echo ctx.generate(model, prompt, charToIx, ixToChar, length = length, temperature = temperature)

if isMainModule:
  main()
