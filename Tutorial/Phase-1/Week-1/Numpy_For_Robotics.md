# NumPy: A Complete Learning Guide

## How to use this guide

This guide moves from the core mechanics of NumPy (arrays, indexing, broadcasting) through to the linear algebra and batch operations that show up constantly in robotics and AI work — transformation matrices, point clouds, forward kinematics. Each section has runnable code. The later sections deliberately reuse the rotation matrix math from earlier so the numbers should look familiar.

Treat this as a working reference alongside your Phase 1 study plan: read a section, run the code, then try the practice progression near the end before moving on.

## 1. Why NumPy exists

A Python list stores a sequence of pointers to separate Python objects scattered in memory. A NumPy array (`ndarray`) stores raw, contiguous, fixed-type data — much closer to how C or a GPU expects numbers to be laid out. That contiguity is what lets NumPy push a loop down into compiled C code instead of running it interpreted, which is where the speed comes from.

```python
import numpy as np

# Python list: element-wise multiply needs an explicit loop
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
result = [x * y for x, y in zip(a, b)]

# NumPy: the loop happens in C, vectorized
a_np = np.array([1, 2, 3, 4])
b_np = np.array([5, 6, 7, 8])
result_np = a_np * b_np  # array([ 5, 12, 21, 32])
```

The mental shift that matters most: stop writing loops over elements, and start thinking in terms of whole-array operations. Almost everything below is in service of that shift.

## 2. Setup

```bash
pip install numpy
```

```python
import numpy as np  # universal convention — always use this alias
```

## 3. The ndarray object

Every array has a handful of properties worth checking reflexively, especially when something doesn't work as expected.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape     # (2, 3) — rows, columns
arr.ndim      # 2 — number of dimensions
arr.size      # 6 — total element count
arr.dtype     # dtype('int64') — element type
arr.itemsize  # 8 — bytes per element
```

`dtype` matters more than it looks. NumPy infers a type from your input, and mixing types (e.g. one float in a list of ints) silently upcasts the whole array. For robotics simulation work, you'll often want to deliberately choose `float32` over the default `float64` for performance when feeding data to GPU-backed tools (Isaac Sim, PyTorch).

```python
np.array([1, 2, 3], dtype=np.float32)
```

## 4. Creating arrays

```python
np.array([1, 2, 3])              # from a list
np.zeros((3, 3))                 # 3x3 of zeros
np.ones((2, 4))                  # 2x4 of ones
np.full((2, 2), 7)                # filled with a constant
np.eye(3)                         # 3x3 identity matrix
np.arange(0, 10, 2)               # [0, 2, 4, 6, 8] — like range()
np.linspace(0, 1, 5)              # 5 evenly spaced points from 0 to 1
np.empty((2, 2))                  # uninitialized memory — fast but garbage values
```

`np.eye(3)` is worth remembering on its own: it's the identity matrix, and it's your starting point any time you build up a transformation matrix piece by piece (see Section 11).

## 5. Indexing and slicing

Slicing works like Python lists but extends across dimensions, with one comma-separated index per axis.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr[0]        # array([1, 2, 3]) — first row
arr[:, 0]     # array([1, 4, 7]) — first column
arr[1, 2]     # 6 — single element
arr[0:2, 1:3] # rows 0-1, columns 1-2
arr[::-1]     # rows reversed
```

Boolean masks and fancy indexing let you select by condition rather than position:

```python
arr[arr > 5]           # array([6, 7, 8, 9]) — flattened, only matches
arr[arr > 5] = 0        # mutate matching elements in place
np.where(arr > 5, 1, 0) # same shape as arr, 1 where condition true, else 0
```

**Pitfall to internalize early**: basic slicing (`arr[0:2]`) returns a *view* — it shares memory with the original array, so modifying the slice modifies the source. Fancy indexing (`arr[[0, 2]]`) and boolean masking return a *copy*. This distinction causes a disproportionate number of confusing bugs.

```python
view = arr[0:2]
view[0, 0] = 999      # this also changes arr!

copy = arr[[0, 1]]
copy[0, 0] = 999      # arr is untouched
```

## 6. Reshaping and combining arrays

```python
arr.reshape(3, 1)          # change shape, same data (errors if sizes don't match)
arr.flatten()              # always returns a copy, 1D
arr.ravel()                # returns a view when possible, 1D — usually preferred
arr.T                      # transpose (swap axes)
arr[:, np.newaxis]         # insert a new axis of size 1
np.expand_dims(arr, axis=0)  # same idea, explicit function form

np.concatenate([a, b], axis=0)  # join along an existing axis
np.vstack([a, b])               # stack as new rows
np.hstack([a, b])               # stack as new columns
np.split(arr, 3)                # divide into 3 equal pieces
```

`reshape(-1, 1)` is a pattern you'll see constantly — the `-1` tells NumPy "figure out this dimension automatically," which is convenient when you want a column vector but don't want to hardcode the length.

## 7. Broadcasting

Broadcasting is the rule set that lets NumPy operate on arrays of different shapes without you writing a loop. Two dimensions are compatible if they're equal, or one of them is 1. NumPy aligns shapes from the trailing (rightmost) dimension and stretches any size-1 dimension to match.

```python
np.array([1, 2, 3]) + 10
# array([11, 12, 13]) — the scalar is broadcast across every element

points = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])          # shape (3, 3): three 3D points
translation = np.array([10, 0, -5])      # shape (3,)

points + translation
# array([[11,  2, -2],
#        [14,  5,  1],
#        [17,  8,  4]])
```

That second example is the everyday robotics case: a batch of N points (shape `(N, 3)`) plus a single translation vector (shape `(3,)`) translates every point at once, no loop required. This is the single most useful trick in the whole library for point-cloud and trajectory work.

## 8. Vectorized math and ufuncs

NumPy's "universal functions" (ufuncs) apply element-wise across an entire array in one call.

```python
np.sqrt(arr)
np.exp(arr)
np.log(arr)
np.sin(arr)        # operates in radians, same as Python's math module
np.clip(arr, 0, 1)   # clamp every value into [0, 1]
arr > 5              # comparison ufuncs return boolean arrays
np.abs(arr)
```

If you ever catch yourself writing `for x in arr: result.append(f(x))`, stop and check whether `f` already has a vectorized ufunc equivalent — it almost always does.

## 9. Aggregations and the axis parameter

```python
arr.sum()          # total sum, all elements
arr.sum(axis=0)    # sum down each column — collapses rows
arr.sum(axis=1)    # sum across each row — collapses columns
arr.mean(axis=0)
arr.std()
arr.min(), arr.max()
arr.argmin(), arr.argmax()   # index of the min/max, not the value itself
```

The intuition for `axis` that actually sticks: the axis you name is the one that *disappears* in the output. `axis=0` walks down rows and collapses them into one row of column sums; `axis=1` walks across columns and collapses them into one column of row sums.

## 10. Linear algebra with `np.linalg` — the robotics core

This is where NumPy connects directly to the rotation matrix math from earlier. The `@` operator (or `np.matmul`) does matrix multiplication; plain `*` would do element-wise multiplication instead, which is a very common mistake.

```python
theta = np.radians(30)
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

point = np.array([4, 3, 0])
rotated = Rz @ point
print(rotated)   # [1.964 4.598 0.   ]  — matches the hand-derived result
```

Other `np.linalg` functions worth knowing on sight:

```python
np.linalg.inv(Rz)        # matrix inverse — for a pure rotation, equals Rz.T
np.linalg.det(Rz)        # determinant — should be 1.0 for a valid rotation
np.linalg.norm(point)    # vector length (magnitude)
np.linalg.eig(Rz)        # eigenvalues and eigenvectors
np.linalg.solve(A, b)    # solves Ax = b directly, faster and more stable than inv(A) @ b
np.cross(v1, v2)         # cross product, e.g. for computing a normal or joint axis
```

A useful sanity check for any rotation matrix you build by hand: `np.linalg.det(R)` should come out to 1.0, and `R @ R.T` should come out to (approximately) the identity matrix. If either fails, something in the construction is wrong.

## 11. Homogeneous transforms — combining rotation and translation

A 4x4 homogeneous transform packs a 3x3 rotation and a 3D translation into one matrix, which is exactly what gets chained together in ROS 2's `tf2` and in forward kinematics.

```python
def homogeneous_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

T = homogeneous_transform(Rz, np.array([1.0, 0.0, 0.5]))

point_h = np.array([4, 3, 0, 1])   # homogeneous coordinate — note the trailing 1
transformed = T @ point_h
print(transformed[:3])   # [2.964 4.598 0.5  ]
```

Chaining transforms is just chaining matrix multiplications: `T_world_base @ T_base_link1 @ T_link1_link2` gives you the combined transform from `link2`'s frame all the way out to the world frame. Order matters — this is the matrix non-commutativity from the 3D rotation discussion showing up again, now at the level of whole transforms rather than individual rotations.

A small forward-kinematics example for a planar 3-joint arm (the same shape of problem as your MuJoCo arm exercise, here worked symbolically instead of simulated):

```python
def link_transform(theta, length):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, length],
        [s,  c, 0],
        [0,  0, 1]
    ])

thetas  = [np.radians(30), np.radians(-45), np.radians(60)]
lengths = [1.0, 0.8, 0.5]

T = np.eye(3)
joint_positions = [T[:2, 2]]
for theta, length in zip(thetas, lengths):
    T = T @ link_transform(theta, length)
    joint_positions.append(T[:2, 2])

print("End-effector position:", T[:2, 2])
```

Each `link_transform` encodes "rotate by this joint angle, then walk forward along the link length." Multiplying them in sequence walks out the kinematic chain from base to end-effector.

## 12. Batch operations — vectorizing over many points or joints

Looping over individual points to rotate them one at a time defeats the purpose of NumPy. Instead, rotate an entire batch in one matrix operation.

```python
rng = np.random.default_rng(0)
points = rng.uniform(-1, 1, size=(1000, 3))   # 1000 random 3D points

rotated_points = points @ Rz.T   # rotates all 1000 points in a single call
```

The `.T` is there because `points` stores each point as a *row*, while the rotation formula was derived for *column* vectors — transposing the rotation matrix reconciles the two layouts without changing the math.

For the more general case — a different rotation matrix for each point, like applying per-joint rotations across a batch of robot configurations — `np.einsum` expresses the batched matrix-vector product directly:

```python
# Rs: shape (N, 3, 3) — one rotation matrix per sample
# points: shape (N, 3) — one point per sample
rotated = np.einsum('nij,nj->ni', Rs, points)
```

`einsum` is the most powerful and least intuitive tool in this list — worth coming back to once the rest feels comfortable, since it generalizes cleanly to the batched tensor operations you'll see in attention mechanisms and transformer layers too.

## 13. Random number generation

NumPy's modern random API is generator-based rather than relying on global state:

```python
rng = np.random.default_rng(seed=42)   # seed for reproducibility
rng.uniform(-1, 1, size=(100, 3))      # uniform random points
rng.normal(0, 0.01, size=(100, 3))     # Gaussian noise — e.g. simulated sensor noise
rng.choice(['x', 'y', 'z'])             # random selection
```

This is the workhorse for domain randomization (randomizing object poses, lighting, physical parameters between simulation runs) and for injecting synthetic sensor noise during testing.

## 14. Performance and memory

A few habits that matter once arrays get large:

```python
arr += 1        # in-place — no new array allocated
arr = arr + 1   # creates a new array — original arr (if referenced elsewhere) is untouched
```

Prefer `float32` over `float64` when feeding data into a GPU pipeline (Isaac Sim, PyTorch) — half the memory, and GPU hardware is often tuned for it. Stick with `float64` (NumPy's default) for general numerical work where precision matters more than throughput, such as one-off linear algebra checks.

Avoid Python-level loops over array elements entirely where possible. If a vectorized form genuinely doesn't exist, `np.vectorize` exists but is mostly a convenience wrapper, not a performance win — it still loops internally. For genuinely loop-bound numerical code, that's a sign to reach for compiled alternatives (Numba, Cython) rather than fighting NumPy.

## 15. Saving and loading data

```python
np.save('points.npy', points)          # single array, binary format
loaded = np.load('points.npy')

np.savez('data.npz', points=points, rotation=Rz)  # multiple named arrays
data = np.load('data.npz')
data['points']

np.savetxt('points.csv', points, delimiter=',')
np.loadtxt('points.csv', delimiter=',')
```

`.npy`/`.npz` are NumPy's own efficient binary formats — use these for anything that stays inside Python. Use `savetxt`/`loadtxt` only when you need a human-readable or cross-tool format like CSV.

## 16. Common pitfalls

| Pitfall | What actually happens |
|---|---|
| `arr * other_arr` when you meant matrix multiplication | Element-wise product, not `@` — silently wrong shape or wrong values |
| Slicing then mutating | Basic slices are views; editing the slice edits the original array too |
| Comparing floats with `==` | Use `np.allclose(a, b)` — floating-point rounding makes exact equality unreliable |
| Forgetting `axis` | Aggregations default to flattening the *entire* array, not just one dimension |
| Shape mismatch in broadcasting | NumPy raises an error rather than guessing — read the shapes in the error message carefully, it tells you exactly which dimensions disagree |
| Integer dtype overflow | `np.array([200], dtype=np.int8) + 100` wraps around silently instead of raising an error |

## 17. Suggested practice progression

Given where Phase 1 is focused (transformation matrices, ROS 2, simulation), a reasonable order to actually build muscle memory:

1. Recreate the 2D and 3D rotation matrix examples above from scratch without looking, for a few different angles, and verify with `np.linalg.det` and `R @ R.T ≈ I`.
2. Rebuild the 3-joint planar arm forward-kinematics example, then extend it to 3D using 4x4 homogeneous transforms instead of 3x3.
3. Generate a random synthetic point cloud (`rng.uniform`, a few thousand points), rotate and translate it as a batch, and compare the bounding box before and after.
4. Re-implement the batch rotation example using `einsum` instead of `@`, applying a *different* rotation to each point, to build comfort with the batched index notation.
5. Cross-check one of your MuJoCo simulation outputs (joint angles, end-effector pose) against a NumPy forward-kinematics calculation done independently — if they agree, you've validated both your simulation setup and your understanding of the transform chain.

## 18. Quick reference

| Category | Functions |
|---|---|
| Creation | `array`, `zeros`, `ones`, `eye`, `arange`, `linspace`, `full`, `empty` |
| Shape | `reshape`, `ravel`, `flatten`, `.T`, `expand_dims`, `concatenate`, `vstack`, `hstack`, `split` |
| Indexing | `arr[i, j]`, boolean masks, fancy indexing, `np.where` |
| Math | `sqrt`, `exp`, `log`, `sin`, `cos`, `clip`, `abs` |
| Aggregation | `sum`, `mean`, `std`, `min`, `max`, `argmin`, `argmax` (all take `axis`) |
| Linear algebra | `@` / `matmul`, `linalg.inv`, `linalg.det`, `linalg.norm`, `linalg.eig`, `linalg.solve`, `cross` |
| Random | `random.default_rng`, `.uniform`, `.normal`, `.choice` |
| I/O | `save`, `load`, `savez`, `savetxt`, `loadtxt` |

## 19. Further resources

The official documentation at numpy.org/doc is genuinely good and worth bookmarking — particularly the "NumPy fundamentals" section for anything not covered here, and the `np.linalg` reference page when you need a function not listed in Section 10. For deeper intuition on broadcasting specifically, searching for visual broadcasting diagrams (several good ones exist showing the shape-stretching mechanics) tends to make the rule click faster than reading it as text.
