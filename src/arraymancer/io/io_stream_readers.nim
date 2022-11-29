# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import streams, endians

proc readInt32BE*(stream: Stream): int32 {.inline.}=
  var raw_bytes = stream.readInt32
  bigEndian32(addr result, addr raw_bytes)

proc readInt64BE*(stream: Stream): int64 {.inline.}=
  var raw_bytes = stream.readInt64
  bigEndian64(addr result, addr raw_bytes)

proc readUInt32BE*(stream: Stream): uint32 {.inline.}=
  var raw_bytes = stream.readUint32
  bigEndian32(addr result, addr raw_bytes)

proc readUInt64BE*(stream: Stream): uint64 {.inline.}=
  var raw_bytes = stream.readUint64
  bigEndian64(addr result, addr raw_bytes)

proc readFloat32BE*(stream: Stream): float32 {.inline.}=
  var raw_bytes = stream.readInt32
  bigEndian32(addr result, addr raw_bytes)

proc readFloat64BE*(stream: Stream): float64 {.inline.}=
  var raw_bytes = stream.readInt64
  bigEndian64(addr result, addr raw_bytes)

proc readInt32LE*(stream: Stream): int32 {.inline.}=
  var raw_bytes = stream.readInt32
  littleEndian32(addr result, addr raw_bytes)

proc readInt64LE*(stream: Stream): int64 {.inline.}=
  var raw_bytes = stream.readInt64
  littleEndian64(addr result, addr raw_bytes)

proc readUInt16LE*(stream: Stream): uint16 {.inline.}=
  var raw_bytes = stream.readUint16
  littleEndian16(addr result, addr raw_bytes)

proc readUInt32LE*(stream: Stream): uint32 {.inline.}=
  var raw_bytes = stream.readUint32
  littleEndian32(addr result, addr raw_bytes)

proc readUInt64LE*(stream: Stream): uint64 {.inline.}=
  var raw_bytes = stream.readUint64
  littleEndian64(addr result, addr raw_bytes)

proc readFloat32LE*(stream: Stream): float32 {.inline.}=
  var raw_bytes = stream.readInt32
  littleEndian32(addr result, addr raw_bytes)

proc readFloat64LE*(stream: Stream): float64 {.inline.}=
  var raw_bytes = stream.readInt64
  littleEndian64(addr result, addr raw_bytes)
