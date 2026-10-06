import
  std / [os, random, times, strformat, algorithm, tables, math, parseopt, strutils],
  ../src/arraymancer

# transformer

proc geglu[T](a: Variable[Tensor[T]]): Variable[Tensor[T]] =
  ## Gated GELU (Shazeer, 2020): split the last dimension in half and gate the
  ## first half with the GELU of the second, xW * GELU(xV).
  let parts = a.chunk(2, axis = a.value.rank - 1)
  parts[0] *. gelu(parts[1])

type
  TransformerBlock[T] = object
    norm1*: RMSNorm[T]
    attn*: MultiHeadAttention[T]
    norm2*: RMSNorm[T]
    mlp_fc1*: Linear[T]
    mlp_fc2*: Linear[T]

  TransformerLM[T] = object
    tok_emb*: Embedding[T]
    pos_emb*: Embedding[T]
    rope*: RotaryEmbedding[T]
    blocks*: seq[TransformerBlock[T]]
    norm_f*: RMSNorm[T]
    head*: Linear[T]
    max_seq_len*: int
    use_rope*: bool

proc initBlock[T](
  ctx: Context[Tensor[T]],
  dim, heads, kvHeads: int,
  mult: int = 4
): TransformerBlock[T] =
  result.norm1 = ctx.init(RMSNorm[T], dim)
  result.attn = ctx.init(MultiHeadAttention[T], dim, heads, dim div heads, kv_heads = kvHeads)
  result.norm2 = ctx.init(RMSNorm[T], dim)
  # geglu: the first projection is doubled, its second half gates the first
  result.mlp_fc1 = ctx.init(Linear[T], dim, 2 * dim * mult)
  result.mlp_fc2 = ctx.init(Linear[T], dim * mult, dim)

proc forward[T](
  self: TransformerBlock[T],
  x: Variable[Tensor[T]],
  rope: RotaryFreqs[T]
): Variable[Tensor[T]] =
  self.forward(x, rope, default(KVCache[T])).output

proc forward[T](
  self: TransformerBlock[T],
  x: Variable[Tensor[T]],
  rope: RotaryFreqs[T],
  past: KVCache[T]
): tuple[output: Variable[Tensor[T]], present: KVCache[T]] =
  # pre-norm attention, continuing from the kv cache
  let (attn_out, present) = self.attn.forward(
    self.norm1.forward(x), is_causal = true, rope = rope, past = past
  )
  let x1 = x + attn_out

  # pre-norm ffn with geglu
  let h = geglu(self.mlp_fc1.forward(self.norm2.forward(x1)))
  result.output = x1 + self.mlp_fc2.forward(h)
  result.present = present

proc initTransformerLM[T](
  ctx: Context[Tensor[T]],
  vocab_size, dim, max_seq_len, layers, heads: int,
  useRope = true,
  rotaryDim = 0,
  kvHeads = 0
): TransformerLM[T] =
  result.tok_emb = ctx.init(Embedding[T], vocab_size, dim)
  if useRope:
    result.rope = RotaryEmbedding[T].init(dim div heads, rotaryDim)
  else:
    result.pos_emb = ctx.init(Embedding[T], max_seq_len, dim)
  result.blocks = newSeq[TransformerBlock[T]](layers)
  for i in 0 ..< layers:
    result.blocks[i] = ctx.initBlock(dim, heads, kvHeads)
  result.norm_f = ctx.init(RMSNorm[T], dim)
  result.head = ctx.init(Linear[T], dim, vocab_size)
  result.max_seq_len = max_seq_len
  result.use_rope = useRope

proc forward[T](self: TransformerLM[T], tokens: Tensor[int]): Variable[Tensor[T]] =
  # empty caches: the cached path is fully differentiable when past is empty
  var caches = newSeq[KVCache[T]](self.blocks.len)
  self.forward(tokens, caches)

proc forward[T](
  self: TransformerLM[T],
  tokens: Tensor[int],
  caches: var seq[KVCache[T]]
): Variable[Tensor[T]] =
  doAssert caches.len == self.blocks.len
  let n = tokens.shape[1]
  let offset = caches[0].past_len
  let rope = if self.use_rope: self.rope.forward(n, offset) else: default(RotaryFreqs[T])

  # token embeddings, plus learned positions continuing from the cached prefix
  var x = self.tok_emb.forward(tokens)
  if not self.use_rope:
    var pos = newTensor[int]([1, n])
    for i in 0 ..< n: pos[0, i] = offset + i
    x = x +. self.pos_emb.forward(pos)

  # transformer blocks, returning the updated kv memories
  for i in 0 ..< self.blocks.len:
    let (output, present) = self.blocks[i].forward(x, rope, caches[i])
    caches[i] = present
    x = output

  # head
  result = self.head.forward(self.norm_f.forward(x))

# sampling

proc sample[T: SomeFloat](probs: Tensor[T], rng: var Rand): int =
  let u = T(rng.rand(1.0))
  var c = 0.T
  for i in 0 ..< probs.size:
    c += probs[i]
    if u <= c: return i
  return probs.size - 1

proc sampleLast[T: SomeFloat](
  logits: Variable[Tensor[T]],
  pos: int,
  temperature: T,
  topK: T,
  rng: var Rand
): int =
  let v = logits.value.shape[^1]
  var last = newTensor[T]([v])
  for i in 0 ..< v:
    last[i] = logits.value[0, pos, i] / temperature

  # top-k filter: keep ceil((1 - topK) * vocab) logits, mask the rest # logits -> filtered
  if topK > 0.T and topK < 1.T:
    let k = max(1, ceil((1.T - topK) * T(v)).int)
    let idx = last.argsort(order = SortOrder.Descending)
    var filtered = newTensor[T]([v])
    for i in 0 ..< v: filtered[i] = T(-Inf)
    for i in 0 ..< k: filtered[idx[i]] = last[idx[i]]
    last = filtered

  result = sample(last.softmax(), rng)

proc generate[T](
  ctx: Context[Tensor[T]],
  model: TransformerLM[T],
  prompt: string,
  charToIx: Table[char, int],
  ixToChar: seq[char],
  length: int = 250,
  temperature: T = 0.7.T,
  topK: T = 0.9.T,
  useCache: bool = false
): string =
  doAssert prompt.len > 0, "prompt must not be empty"
  var rng = initRand(42)
  var tokens = newSeq[int]()
  for ch in prompt:
    tokens.add(if ch in charToIx: charToIx[ch] else: 0)

  result = ""

  ctx.no_grad_mode:
    var caches = newSeq[KVCache[T]](if useCache: model.blocks.len else: 0)

    for _ in 0 ..< length:
      let total = tokens.len
      # full context for rope, a sliding max_seq_len window otherwise
      let window = if model.use_rope: total else: min(total, model.max_seq_len)

      # a cache that does not hold the current window prefix is invalid
      if useCache and caches[0].past_len != window - 1:
        for c in caches.mitems: c = default(KVCache[T])

      # feed the window, or just the last token when continuing from a cache
      let n = if useCache and not caches[0].isEmpty: 1 else: window
      if useCache and not caches[0].isEmpty:
        doAssert caches[0].past_len == window - 1

      let offset = total - n
      var inp = newTensor[int]([1, n])
      for i in 0 ..< n:
        inp[0, i] = tokens[offset + i]

      let logits =
        if useCache: model.forward(inp, caches)
        else: model.forward(inp)

      let next_id = sampleLast(logits, n - 1, temperature, topK, rng)
      tokens.add next_id
      result.add ixToChar[next_id]

# main

proc main() =
  var
    steps = 2000
    length = 350
    temperature = 0.65'f32
    topK = 0.9'f32
    prompt = "ROMEO:\n"
    sampleEvery = 100
    kvCache = true
    compare = false
    useRope = true

  for kind, key, val in getopt():
    case kind
    of cmdLongOption, cmdShortOption:
      case key
      of "steps": steps = parseInt(val)
      of "length": length = parseInt(val)
      of "temperature": temperature = parseFloat(val).float32
      of "top-k": topK = parseFloat(val).float32
      of "prompt": prompt = val
      of "sample-every": sampleEvery = parseInt(val)
      of "kv-cache": kvCache = if val.len == 0: true else: parseBool(val)
      of "compare": compare = true
      of "rope": useRope = if val.len == 0: true else: parseBool(val)
      of "abs-pos": useRope = false
      else: discard
    else: discard

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
    kv_heads = heads div 2 # grouped query attention: half the heads for keys/values
    layers = 3
    seq_len = 64
    batch_size = 32
    lr = 0.0015'f32
    head_dim = dim div heads
    rotary_dim = head_dim div 2 # partial rotary

  let ropeInfo = if useRope: &", rope, rotary_dim={rotary_dim}" else: ", abs pos"
  echo &"Shakespeare ({text.len} chars, vocab {vocab_size}) | dim={dim}, heads={heads}, kv_heads={kv_heads}, layers={layers}, ctx={seq_len}{ropeInfo}"

  let ctx = newContext Tensor[float32]
  let model = ctx.initTransformerLM(vocab_size, dim, seq_len, layers, heads, useRope = useRope, rotaryDim = rotary_dim, kvHeads = kv_heads)
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
      echo &"--- sample @ step {step} | kv-cache: {kvCache} ---"
      echo &"prompt: {prompt.escape}"
      echo "--- generated ---"
      echo ctx.generate(model, prompt, charToIx, ixToChar, length = min(150, length), temperature = temperature, topK = topK, useCache = kvCache)

  echo &"\ntrained in {epochTime() - t0:.1f}s\n"

  # generate
  echo "--- Generated Shakespeare ---"
  echo &"prompt: {prompt.escape}"

  if compare:
    var
      outputs: seq[string] = @[]
      times: seq[float] = @[]
    for mode in [false, true]:
      let t1 = epochTime()
      outputs.add ctx.generate(model, prompt, charToIx, ixToChar, length = length, temperature = temperature, topK = topK, useCache = mode)
      times.add epochTime() - t1
    echo &"--- no-cache {times[0]:.2f}s | kv-cache {times[1]:.2f}s | identical: {outputs[0] == outputs[1]} ---"
    echo outputs[1]
  else:
    let t1 = epochTime()
    let generated = ctx.generate(model, prompt, charToIx, ixToChar, length = length, temperature = temperature, topK = topK, useCache = kvCache)
    echo &"--- kv-cache: {kvCache} | {epochTime() - t1:.2f}s | {generated.len} chars ---"
    echo generated

if isMainModule:
  main()
