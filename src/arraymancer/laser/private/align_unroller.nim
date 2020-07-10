# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

func round_step_down*(x: Natural, step: static Natural): int {.inline.} =
  ## Round the input to the previous multiple of "step"
  when (step and (step - 1)) == 0:
    # Step is a power of 2. (If compiler cannot prove that x>0 it does not make the optim)
    result = x and not(step - 1)
  else:
    result = x - x mod step

func round_step_up*(x: Natural, step: static Natural): int {.inline.} =
  ## Round the input to the next multiple of "step"
  when (step and (step - 1)) == 0:
    # Step is a power of 2. (If compiler cannot prove that x>0 it does not make the optim)
    result = (x + step - 1) and not(step - 1)
  else:
    result = ((x + step - 1) div step) * step

when isMainModule:
  doAssert round_step_up(10, 4) == 12
  doAssert round_step_up(10, 8) == 16
  doAssert round_step_up(65, 64) == 128
  doAssert round_step_up(1, 3) == 3
  doAssert round_step_up(19, 24) == 24
  doAssert round_step_up(8, 4) == 8
  doAssert round_step_up(64, 64) == 64
  doAssert round_step_up(24, 24) == 24
  doAssert round_step_up(3, 3) == 3

  doAssert round_step_down(10, 4) == 8
  doAssert round_step_down(10, 8) == 8
  doAssert round_step_down(65, 64) == 64
  doAssert round_step_down(1, 3) == 0
  doAssert round_step_down(19, 24) == 0
  doAssert round_step_down(8, 4) == 8
  doAssert round_step_down(64, 64) == 64
  doAssert round_step_down(24, 24) == 24
  doAssert round_step_down(3, 3) == 3
