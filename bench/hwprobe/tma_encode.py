import ctypes
from ctypes import POINTER, byref, c_char_p, c_int, c_size_t, c_uint, c_uint32, c_uint64, c_void_p
import json
import sys

TILES, WIDTH, HEIGHT = 64, 64, 64
WORDS = TILES * WIDTH * HEIGHT
UINT32, L2_256B = 2, 3


def load_driver():
    lib = ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [c_uint], "cuDeviceGet": [POINTER(c_int), c_int], "cuDevicePrimaryCtxRetain": [POINTER(c_void_p), c_int],
        "cuCtxSetCurrent": [c_void_p], "cuModuleLoad": [POINTER(c_void_p), c_char_p],
        "cuModuleGetFunction": [POINTER(c_void_p), c_void_p, c_char_p], "cuMemAlloc_v2": [POINTER(c_uint64), c_size_t],
        "cuMemcpyHtoD_v2": [c_uint64, c_void_p, c_size_t], "cuMemcpyDtoH_v2": [c_void_p, c_uint64, c_size_t],
        "cuMemsetD32_v2": [c_uint64, c_uint, c_size_t], "cuCtxSynchronize": [], "cuGetErrorString": [c_int, POINTER(c_char_p)],
        "cuLaunchKernel": [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, POINTER(c_void_p), c_void_p],
        "cuTensorMapEncodeTiled": [c_void_p, c_int, c_uint32, c_void_p, POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint32),
                                   POINTER(c_uint32), c_int, c_int, c_int, c_int],
    }
    for name, argtypes in signatures.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = c_int
    return lib


def check(lib, result, what):
    if result:
        text = c_char_p()
        lib.cuGetErrorString(result, byref(text))
        raise RuntimeError(f"{what}: {text.value.decode() if text.value else result}")


def aligned_map():
    storage = ctypes.create_string_buffer(192)
    return storage, ctypes.addressof(storage) + (-ctypes.addressof(storage)) % 64


def encode_ctypes(lib, address):
    storage, pointer = aligned_map()
    dims = (c_uint64 * 2)(WIDTH, HEIGHT * TILES)
    strides = (c_uint64 * 1)(WIDTH * 4)
    box = (c_uint32 * 2)(WIDTH, HEIGHT)
    steps = (c_uint32 * 2)(1, 1)
    check(lib, lib.cuTensorMapEncodeTiled(pointer, UINT32, 2, address, dims, strides, box, steps, 0, 0, L2_256B, 0), "ctypes cuTensorMapEncodeTiled")
    return ctypes.string_at(pointer, 128)


def cuda_python():
    try:
        from cuda.bindings import driver
        return driver, "cuda.bindings.driver"
    except ImportError:
        pass
    try:
        from cuda import cuda as driver
        return driver, "cuda.cuda"
    except ImportError:
        return None, None


def encode_cuda_python(driver, address):
    u64, u32 = driver.cuuint64_t, driver.cuuint32_t
    result, tensor_map = driver.cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, address,
        [u64(WIDTH), u64(HEIGHT * TILES)], [u64(WIDTH * 4)], [u32(WIDTH), u32(HEIGHT)], [u32(1), u32(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B, driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuda-python cuTensorMapEncodeTiled: {result}")
    if hasattr(tensor_map, "getPtr"):
        return ctypes.string_at(tensor_map.getPtr(), 128)
    return b"".join(int(word).to_bytes(8, "little") for word in tensor_map.opaque)


def run(lib, function, blob, destination, expected, repeats):
    storage, pointer = aligned_map()
    ctypes.memmove(pointer, blob, 128)
    output = c_uint64(destination)
    params = (c_void_p * 2)(pointer, ctypes.addressof(output))
    observed = (c_uint32 * WORDS)()
    for _ in range(repeats):
        check(lib, lib.cuMemsetD32_v2(destination, 0, WORDS), "cuMemsetD32")
        check(lib, lib.cuLaunchKernel(function, TILES, 1, 1, 128, 1, 1, 0, None, params, None), "cuLaunchKernel")
        check(lib, lib.cuCtxSynchronize(), "cuCtxSynchronize")
        check(lib, lib.cuMemcpyDtoH_v2(observed, destination, WORDS * 4), "cuMemcpyDtoH")
        if bytes(observed) != expected:
            raise RuntimeError("TMA load through host-encoded descriptor returned wrong data")


def main():
    cubin = sys.argv[1]
    repeats = max(1, min(16, int(sys.argv[2]) if len(sys.argv) > 2 else 4))
    lib = load_driver()
    check(lib, lib.cuInit(0), "cuInit")
    device, context, module, function = c_int(), c_void_p(), c_void_p(), c_void_p()
    check(lib, lib.cuDeviceGet(byref(device), 0), "cuDeviceGet")
    check(lib, lib.cuDevicePrimaryCtxRetain(byref(context), device), "cuDevicePrimaryCtxRetain")
    check(lib, lib.cuCtxSetCurrent(context), "cuCtxSetCurrent")
    check(lib, lib.cuModuleLoad(byref(module), cubin.encode()), "cuModuleLoad")
    check(lib, lib.cuModuleGetFunction(byref(function), module, b"tma_encode_kernel"), "cuModuleGetFunction")
    host = (c_uint32 * WORDS)(*((i * 1664525 + 1013904223) & 0xFFFFFFFF for i in range(WORDS)))
    source, destination = c_uint64(), c_uint64()
    check(lib, lib.cuMemAlloc_v2(byref(source), WORDS * 4), "cuMemAlloc")
    check(lib, lib.cuMemAlloc_v2(byref(destination), WORDS * 4), "cuMemAlloc")
    check(lib, lib.cuMemcpyHtoD_v2(source.value, host, WORDS * 4), "cuMemcpyHtoD")
    expected = bytes(host)
    blobs = {"ctypes": encode_ctypes(lib, source.value)}
    driver, module_name = cuda_python()
    if driver is None:
        print("cuda-python: not installed; ctypes backend only", file=sys.stderr)
    else:
        blobs["cuda-python"] = encode_cuda_python(driver, source.value)
        same = blobs["cuda-python"] == blobs["ctypes"]
        print(f"cuda-python ({module_name}) descriptor bytes {'identical to' if same else 'differ from'} ctypes", file=sys.stderr)
    for name, blob in blobs.items():
        run(lib, function.value, blob, destination.value, expected, repeats)
        print(f"{name}: encoded descriptor passed as __grid_constant__ param; {TILES} tiles x {repeats} launches validated", file=sys.stderr)
    print(json.dumps({"metric": "validated_encoder_backends", "value": len(blobs), "unit": "backends"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
