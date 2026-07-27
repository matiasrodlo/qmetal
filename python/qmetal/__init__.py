"""Python bindings for qmetal.

A ctypes wrapper over the C ABI, so installing needs no compiler, no nanobind,
no pybind11 -- just the shared library next to it. Amplitudes come back as a
zero-copy numpy view of the unified-memory buffer when numpy is present.

    import qmetal
    c = qmetal.Circuit(4)
    c.h(0)
    for i in range(3):
        c.cx(i, i + 1)
    sim = qmetal.Simulator(4)
    sim.run(c)
    print(sim.counts(1000))
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER, c_char_p, c_double, c_float, c_int, c_size_t, c_uint32, c_uint64,
    c_void_p, byref,
)

__all__ = ["Circuit", "Simulator", "Hamiltonian", "QMetalError", "version",
           "is_available", "device_name", "max_qubits"]


class QMetalError(RuntimeError):
    """Raised when the native library reports a failure."""


def _find_library() -> str:
    env = os.environ.get("QMETAL_LIBRARY")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "libqmetal.dylib"),
        os.path.join(here, "..", "..", "build", "libqmetal.dylib"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.normpath(path)
    raise QMetalError(
        "libqmetal.dylib not found. Build it with `make lib`, or point "
        "QMETAL_LIBRARY at it."
    )


if sys.platform != "darwin":
    raise QMetalError("qmetal is Apple Silicon only")

_lib = ctypes.CDLL(_find_library())


def _decl(name, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


_u64p, _f64p, _u32p = POINTER(c_uint64), POINTER(c_double), POINTER(c_uint32)


class _Hamiltonian(ctypes.Structure):
    _fields_ = [
        ("num_terms", c_uint32),
        ("coeffs", _f64p),
        ("x_masks", _u64p),
        ("y_masks", _u64p),
        ("z_masks", _u64p),
    ]


_version = _decl("qm_version", None, POINTER(c_int), POINTER(c_int), POINTER(c_int))
_last_error = _decl("qm_last_error", c_char_p)
_is_available = _decl("qm_is_available", c_int)
_device_name = _decl("qm_device_name", c_char_p)
_max_qubits = _decl("qm_max_qubits", c_uint32)

_circuit_create = _decl("qm_circuit_create", c_void_p, c_uint32)
_circuit_destroy = _decl("qm_circuit_destroy", None, c_void_p)
_circuit_add = _decl("qm_circuit_add", c_int, c_void_p, c_int, _u32p, c_uint32,
                     _f64p, c_uint32)
_circuit_size = _decl("qm_circuit_size", c_size_t, c_void_p)
_plan_size = _decl("qm_circuit_plan_size", c_size_t, c_void_p, c_int)

_create = _decl("qm_create", c_void_p, c_uint32)
_destroy = _decl("qm_destroy", None, c_void_p)
_reset = _decl("qm_reset", c_int, c_void_p)
_run = _decl("qm_run", c_int, c_void_p, c_void_p)
_run_unfused = _decl("qm_run_unfused", c_int, c_void_p, c_void_p)
_sync = _decl("qm_synchronize", c_int, c_void_p)
_num_qubits = _decl("qm_num_qubits", c_uint32, c_void_p)
_dim = _decl("qm_dim", c_uint64, c_void_p)
_dispatch_count = _decl("qm_dispatch_count", c_uint64, c_void_p)
_amplitudes = _decl("qm_amplitudes", POINTER(c_float), c_void_p)
_norm = _decl("qm_norm", c_int, c_void_p, _f64p)
_expect_pauli = _decl("qm_expectation_pauli", c_int, c_void_p, c_uint64,
                      c_uint64, c_uint64, _f64p)
_expect = _decl("qm_expectation", c_int, c_void_p, POINTER(_Hamiltonian), _f64p)
_sample = _decl("qm_sample", c_int, c_void_p, c_size_t, c_uint64, _u64p)
_energy_grad = _decl("qm_energy_and_gradient", c_int, c_void_p, c_void_p,
                     POINTER(_Hamiltonian), _f64p, _f64p)


def _check(rc: int) -> None:
    if rc != 0:
        raise QMetalError(_last_error().decode("utf-8", "replace"))


# Gate codes: must match qm_gate_kind in qmetal.h.
_X, _Y, _Z, _H, _S, _SDG, _T, _TDG = range(8)
_RX, _RY, _RZ, _P = 8, 9, 10, 11
_CX, _CY, _CZ, _SWAP, _CP, _CCX = 12, 13, 14, 15, 16, 17


def version() -> tuple[int, int, int]:
    a, b, c = c_int(), c_int(), c_int()
    _version(byref(a), byref(b), byref(c))
    return a.value, b.value, c.value


def is_available() -> bool:
    return bool(_is_available())


def device_name() -> str:
    return _device_name().decode()


def max_qubits() -> int:
    return int(_max_qubits())


class Hamiltonian:
    """A real-weighted sum of Pauli strings, given as bitmasks."""

    def __init__(self):
        self.terms: list[tuple[float, int, int, int]] = []

    def add(self, coeff: float, x_mask: int = 0, y_mask: int = 0,
            z_mask: int = 0) -> "Hamiltonian":
        self.terms.append((float(coeff), int(x_mask), int(y_mask), int(z_mask)))
        return self

    def add_string(self, coeff: float, s: str) -> "Hamiltonian":
        """`s` is "IXYZ"-style, leftmost character is qubit 0."""
        x = y = z = 0
        for q, ch in enumerate(s.upper()):
            bit = 1 << q
            if ch == "X":
                x |= bit
            elif ch == "Y":
                y |= bit
            elif ch == "Z":
                z |= bit
            elif ch != "I":
                raise ValueError(f"bad Pauli character {ch!r}")
        return self.add(coeff, x, y, z)

    def _c(self):
        n = len(self.terms)
        coeffs = (c_double * n)(*[t[0] for t in self.terms])
        xs = (c_uint64 * n)(*[t[1] for t in self.terms])
        ys = (c_uint64 * n)(*[t[2] for t in self.terms])
        zs = (c_uint64 * n)(*[t[3] for t in self.terms])
        h = _Hamiltonian(n, coeffs, xs, ys, zs)
        # The arrays must outlive the struct, so keep them alive on it.
        h._keep = (coeffs, xs, ys, zs)
        return h


class Circuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = int(num_qubits)
        self._h = _circuit_create(self.num_qubits)
        if not self._h:
            raise QMetalError(_last_error().decode())

    def __del__(self):
        if getattr(self, "_h", None):
            _circuit_destroy(self._h)
            self._h = None

    def __len__(self) -> int:
        return int(_circuit_size(self._h))

    def _add(self, kind, qubits, params=()):
        qs = (c_uint32 * len(qubits))(*qubits)
        ps = (c_double * len(params))(*params) if params else None
        _check(_circuit_add(self._h, kind, qs, len(qubits),
                            ps, len(params)))
        return self

    # Single-qubit
    def x(self, q): return self._add(_X, (q,))
    def y(self, q): return self._add(_Y, (q,))
    def z(self, q): return self._add(_Z, (q,))
    def h(self, q): return self._add(_H, (q,))
    def s(self, q): return self._add(_S, (q,))
    def sdg(self, q): return self._add(_SDG, (q,))
    def t(self, q): return self._add(_T, (q,))
    def tdg(self, q): return self._add(_TDG, (q,))
    def rx(self, q, theta): return self._add(_RX, (q,), (theta,))
    def ry(self, q, theta): return self._add(_RY, (q,), (theta,))
    def rz(self, q, theta): return self._add(_RZ, (q,), (theta,))
    def p(self, q, theta): return self._add(_P, (q,), (theta,))

    # Multi-qubit
    def cx(self, c, t): return self._add(_CX, (c, t))
    def cy(self, c, t): return self._add(_CY, (c, t))
    def cz(self, a, b): return self._add(_CZ, (a, b))
    def swap(self, a, b): return self._add(_SWAP, (a, b))
    def cp(self, a, b, theta): return self._add(_CP, (a, b), (theta,))
    def ccx(self, c0, c1, t): return self._add(_CCX, (c0, c1, t))

    def plan_size(self, fuse: bool = True) -> int:
        """GPU passes this circuit lowers to -- what execution actually costs."""
        return int(_plan_size(self._h, 1 if fuse else 0))


class Simulator:
    def __init__(self, num_qubits: int):
        self._h = _create(int(num_qubits))
        if not self._h:
            raise QMetalError(_last_error().decode())

    def __del__(self):
        if getattr(self, "_h", None):
            _destroy(self._h)
            self._h = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.__del__()

    @property
    def num_qubits(self) -> int: return int(_num_qubits(self._h))
    @property
    def dim(self) -> int: return int(_dim(self._h))
    @property
    def dispatch_count(self) -> int: return int(_dispatch_count(self._h))

    def reset(self): _check(_reset(self._h)); return self

    def run(self, circuit: Circuit, fuse: bool = True):
        _check((_run if fuse else _run_unfused)(self._h, circuit._h))
        return self

    def synchronize(self): _check(_sync(self._h)); return self

    def amplitudes(self):
        """Zero-copy numpy view of the state, or a list if numpy is absent."""
        p = _amplitudes(self._h)
        if not p:
            raise QMetalError(_last_error().decode())
        n = self.dim
        try:
            import numpy as np
        except ImportError:
            return [complex(p[2 * i], p[2 * i + 1]) for i in range(n)]
        buf = np.ctypeslib.as_array(p, shape=(2 * n,))
        return buf.view(np.complex64)   # no copy: unified memory

    def norm(self) -> float:
        out = c_double()
        _check(_norm(self._h, byref(out)))
        return out.value

    def expectation(self, observable) -> float:
        out = c_double()
        if isinstance(observable, Hamiltonian):
            h = observable._c()
            _check(_expect(self._h, byref(h), byref(out)))
        elif isinstance(observable, str):
            h = Hamiltonian().add_string(1.0, observable)._c()
            _check(_expect(self._h, byref(h), byref(out)))
        else:
            raise TypeError("observable must be a Hamiltonian or a Pauli string")
        return out.value

    def sample(self, shots: int, seed: int = 0):
        buf = (c_uint64 * int(shots))()
        _check(_sample(self._h, int(shots), int(seed), buf))
        return list(buf)

    def counts(self, shots: int, seed: int = 0) -> dict[str, int]:
        """Bitstring counts, little-endian: qubit 0 is the rightmost character."""
        n = self.num_qubits
        out: dict[str, int] = {}
        for v in self.sample(shots, seed):
            key = format(v, f"0{n}b")
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def energy_and_gradient(self, circuit: Circuit, hamiltonian: Hamiltonian):
        """Adjoint gradient: one entry per gate, zero where unparameterised."""
        h = hamiltonian._c()
        energy = c_double()
        grad = (c_double * len(circuit))()
        _check(_energy_grad(self._h, circuit._h, byref(h), byref(energy), grad))
        return energy.value, list(grad)
