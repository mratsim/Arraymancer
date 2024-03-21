import std / times

type FooBase = ref object {.inheritable.}
  dummy: int

type Foo{.final.} = ref object of FooBase
  value : float32


proc inplace_add_proc(x: var Foo, a: float32) =
  x.value += a

proc inplace_add_closure(x: var float32, a: float32) =
  proc add_closure(v: var float32) = v += a
  add_closure(x)

method inplace_add_method(x: FooBase, a: float32) {.base.} =
  discard

method inplace_add_method(x: Foo, a: float32) =
  x.value += a

var bar : Foo
new bar
var start = cpuTime()
for i in 0..<100000000:
  inplace_add_proc(bar, 1.0f)
echo " Proc with ref object ", cpuTime() - start

var x : float32
start = cpuTime()
for i in 0..<100000000:
  inplace_add_closure(x, 1.0f)
echo " Closures ", cpuTime() - start

var baz : Foo
new baz
start = cpuTime()
for i in 0..<100000000:
  inplace_add_method(baz, 1.0f)
echo " Methods ", cpuTime() - start

# Results with -d:release on i5-5257U (dual-core mobile 2.7GHz, turbo 3.1)
# Proc with ref object 0.099993
# Closures 2.708598
# Methods 0.3122219999999998