from complex import Complex, complex32, complex64

template numberOne*(T:type SomeNumber):SomeNumber= T(1)
template numberOne*(T:type Complex[float32]):Complex[float32] = complex32(1.0, 0.0)
template numberOne*(T:type Complex[float64]):Complex[float64] = complex64(1.0, 0.0)

converter Complex64*[T:SomeNumber](x: T):Complex[float64]=
  result.re = x.float64
  result.im = 0'f64

converter Complex32*[T:SomeNumber](x: T):Complex[float32]=
  result.re = x.float32
  result.im = 0'f32

