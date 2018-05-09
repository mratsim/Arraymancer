# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import streams, endians

proc readInt32BE*(stream: FileStream): int32 {.inline.}=
  var raw_bytes = stream.readInt32
  bigEndian32(addr result, addr raw_bytes)

proc readInt64BE*(stream: FileStream): int64 {.inline.}=
  var raw_bytes = stream.readInt64
  bigEndian64(addr result, addr raw_bytes)

proc readUInt32BE*(stream: FileStream): uint32 {.inline.}=
  var raw_bytes = stream.readUInt32
  bigEndian32(addr result, addr raw_bytes)

proc readUInt64BE*(stream: FileStream): uint64 {.inline.}=
  var raw_bytes = stream.readUInt64
  bigEndian64(addr result, addr raw_bytes)

proc readFloat32BE*(stream: FileStream): float32 {.inline.}=
  var raw_bytes = stream.readInt32
  bigEndian32(addr result, addr raw_bytes)

proc readFloat64BE*(stream: FileStream): float64 {.inline.}=
  var raw_bytes = stream.readInt64
  bigEndian64(addr result, addr raw_bytes)

proc readInt32LE*(stream: FileStream): int32 {.inline.}=
  var raw_bytes = stream.readInt32
  littleEndian32(addr result, addr raw_bytes)

proc readInt64LE*(stream: FileStream): int64 {.inline.}=
  var raw_bytes = stream.readInt64
  littleEndian64(addr result, addr raw_bytes)

proc readUInt16LE*(stream: FileStream): uint16 {.inline.}=
  var raw_bytes = stream.readUInt16
  littleEndian16(addr result, addr raw_bytes)

proc readUInt32LE*(stream: FileStream): uint32 {.inline.}=
  var raw_bytes = stream.readUInt32
  littleEndian32(addr result, addr raw_bytes)

proc readUInt64LE*(stream: FileStream): uint64 {.inline.}=
  var raw_bytes = stream.readUInt64
  littleEndian64(addr result, addr raw_bytes)

proc readFloat32LE*(stream: FileStream): float32 {.inline.}=
  var raw_bytes = stream.readInt32
  littleEndian32(addr result, addr raw_bytes)

proc readFloat64LE*(stream: FileStream): float64 {.inline.}=
  var raw_bytes = stream.readInt64
  littleEndian64(addr result, addr raw_bytes)
