"""
LAYOUT ALGEBRA PLAYGROUND  (CPU-side, no kernel launch)

How to use this file:
  1. For each exercise, WRITE your predicted answer in the "PREDICT:" line
     BEFORE running. Do the arithmetic by hand (coord * stride, summed).
  2. Run:  python3 layouts_playground.py
  3. Compare the printed value to your prediction. If they differ, redo the
     hand calculation until you see why.

This drills the 8 concepts from cute_algebra.md. Layout(coord) = offset is the
whole game; everything else is layouts built from layouts.
"""

from cutlass import cute


@cute.jit
def playground():
    # ------------------------------------------------------------------
    # EX1  Layout = coordinate -> offset   (row-major 3x4, stride (4,1))
    #   offset(i,j) = i*4 + j
    # PREDICT: L1(1,2) = ____    L1(2,3) = ____
    L1 = cute.make_layout((3, 4), stride=(4, 1))
    cute.printf("EX1  L1(1,2)=%d  L1(2,3)=%d\n", L1((1, 2)), L1((2, 3)))

    # ------------------------------------------------------------------
    # EX2  Shape defines the coordinate space; size = product of extents
    # PREDICT: size of shape ((2,3),4) = ____
    cute.printf("EX2  size((2,3),4)=%d\n", cute.size(((2, 3), 4)))

    # ------------------------------------------------------------------
    # EX3  Stride = memory movement   (COLUMN-major 4x8, stride (1,4))
    #   offset(i,j) = i*1 + j*4
    # PREDICT: L3(2,0)=____  L3(0,1)=____  L3(3,7)=____
    L3 = cute.make_layout((4, 8), stride=(1, 4))
    cute.printf("EX3  L3(2,0)=%d  L3(0,1)=%d  L3(3,7)=%d\n",
                L3((2, 0)), L3((0, 1)), L3((3, 7)))

    # ------------------------------------------------------------------
    # EX4  Hierarchical shape = tiles. 128x128 matrix, 8x8 tiles, row-major.
    #   shape ((16,8),(16,8))  stride ((8*128,1*1)?...) -- here we use the
    #   row-major global strides: outer-row jumps 8 rows (8*128), inner-row 128,
    #   outer-col jumps 8 cols (8), inner-col 1.
    # coordinate (tile_r, inner_r), (tile_c, inner_c) = (2,4),(3,5)
    # PREDICT: global offset = (2*8+4)*128 + (3*8+5) = ____
    L4 = cute.make_layout(((16, 8), (16, 8)), stride=((8 * 128, 128), (8, 1)))
    cute.printf("EX4  L4((2,4),(3,5))=%d\n", L4(((2, 4), (3, 5))))

    # ------------------------------------------------------------------
    # EX5  Composition: feed L2 coords into L1's mapping.
    #   L1=(8,8):(8,1) row-major;  L2=(2,2):(4,1)
    # PREDICT: comp(1,0)=____  comp(0,1)=____  comp(1,1)=____
    cL1 = cute.make_layout((8, 8), stride=(8, 1))
    cL2 = cute.make_layout((2, 2), stride=(4, 1))
    comp = cute.composition(cL1, cL2)
    cute.printf("EX5  comp(1,0)=%d  comp(0,1)=%d  comp(1,1)=%d\n",
                comp((1, 0)), comp((0, 1)), comp((1, 1)))

    # ------------------------------------------------------------------
    # EX6  logical_divide: split 12:1 into tiles of 4.
    #   result coord is (inner, tile): div(inner, tile) = tile*4 + inner
    # PREDICT: div(0,0)=____  div(1,0)=____  div(0,1)=____  div(0,2)=____
    div = cute.logical_divide(cute.make_layout(12, stride=1), 4)
    cute.printf("EX6  div(0,0)=%d  div(1,0)=%d  div(0,1)=%d  div(0,2)=%d\n",
                div((0, 0)), div((1, 0)), div((0, 1)), div((0, 2)))

    # ------------------------------------------------------------------
    # EX7  coalesce: ((4,2),8) contiguous collapses, offsets UNCHANGED.
    #   Prove it: original L7(i) must equal coalesced cL7(i) for all i.
    # PREDICT: do L7(5) and cL7(5) match? (yes/no) ____
    L7 = cute.make_layout(((4, 2), 8), stride=((1, 4), 8))
    cL7 = cute.coalesce(L7)
    cute.printf("EX7  size=%d  cL7(5)=%d (compare to your hand calc)\n",
                cute.size(cL7), cL7(5))

    # ------------------------------------------------------------------
    # EX8  complement: 4:1 under 16. The 4 covered, complement sweeps the rest.
    #   complement has size 16/4 = 4, stride 4.
    # PREDICT: complement size=____  cmp(1)=____  cmp(2)=____
    cmp = cute.complement(cute.make_layout(4, stride=1), 16)
    cute.printf("EX8  cmp size=%d  cmp(1)=%d  cmp(2)=%d\n",
                cute.size(cmp), cmp(1), cmp(2))


playground()
print("done -- compare every printed value to your PREDICT lines")
