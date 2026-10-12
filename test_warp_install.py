"""Quick smoke test for the NVIDIA Warp installation."""
import warp as wp


@wp.kernel
def add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = a[i] + b[i]


def main() -> None:
    print("warp version:", wp.__version__)
    device = "cuda:0"
    a = wp.array([1.0, 2.0, 3.0], dtype=float, device=device)
    b = wp.array([10.0, 20.0, 30.0], dtype=float, device=device)
    c = wp.empty(3, dtype=float, device=device)
    wp.launch(add, dim=3, inputs=[a, b, c], device=device)
    wp.synchronize()
    print("GPU kernel result:", c.list())


if __name__ == "__main__":
    main()
