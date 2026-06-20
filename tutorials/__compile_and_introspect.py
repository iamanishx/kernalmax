import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.layout import LayoutEnum

import cupy as cp
import numpy as np


@cute.jit
def introspect(t: cute.Tensor, label: cutlass.Constexpr):
    cute.printf(label)
    cute.printf("  max_alignment        = %d bytes", t.layout.max_alignment)
    cute.printf("  element_type.width   = %d bits", t.element_type.width)
    cute.printf("  size_in_bytes(layout)= %d", cute.size_in_bytes(t.element_type, t.layout))


def main():
    print("Two 128x64 fp32 tensors, one ROW-major and one COL-major.\n")

    x_np = np.random.randn(128, 64).astype(np.float32)
    t_row = from_dlpack(cp.asarray(x_np))                       # row-major (numpy default)
    t_col = from_dlpack(cp.asarray(np.asfortranarray(x_np)))    # col-major

    print(f"row tensor: LayoutEnum.from_tensor = {LayoutEnum.from_tensor(t_row)}")
    print(f"col tensor: LayoutEnum.from_tensor = {LayoutEnum.from_tensor(t_col)}")
    print()

    introspect(t_row, "[ROW]")
    introspect(t_col, "[COL]")
    cp.cuda.get_current_stream().synchronize()

    print("\nHow mm.py uses these:")
    print("  - LayoutEnum.from_tensor  -> branch on row vs col for the threading pattern")
    print("  - max_alignment           -> if % 16 == 0 use 128-bit copies, else scalar")
    print("  - element_type.width      -> compute num_bits_per_copy")
    print("  - size_in_bytes           -> pass as smem=... in the launch")


if __name__ == "__main__":
    main()
