Nice, this is going to turn into a really solid little robotics core.

Below is a **detailed + exhaustive implementation checklist** for the library, organized in phases. You can treat Phase 0–1 as today’s focus, then keep going over the week.

I’ll use `[ ]` checkboxes so you can literally copy-paste this into your README / notes and tick things off.

---

## 📦 Phase 0 – Repo & Project Scaffolding

### 0.1. Repository layout

**[ ]** Create repo and base structure:

```text
robotics_lib/
    robotics_lib/
        __init__.py
        geometry.py
        kinematics.py
        ik.py
        dynamics.py
        models.py
        utils.py
    tests/
        __init__.py
        test_smoke.py
    README.md
    CONVENTIONS.md
    pyproject.toml or setup.cfg / setup.py  (optional for now)
    .gitignore
```

**Checklist:**

* [ ] `robotics_lib/__init__.py` created

  * [ ] Expose top-level symbols you want (optionally) e.g. `from .models import planar_2link`
* [ ] `tests/` package created with `__init__.py`
* [ ] `.gitignore` includes at least: `__pycache__/`, `.pytest_cache/`, `.vscode/`, `*.pyc`

---

## 📘 Phase 1 – Documentation: README + CONVENTIONS

### 1.1. README.md

**Goal:** If a stranger clones your repo, they can understand what the library does and how to use it in under 2 minutes.

**Content checklist:**

* [ ] Project name + one-line description (e.g. “A minimal Python robotics core (FK, IK, dynamics) for learning & interviews.”)
* [ ] Motivation:

  * [ ] Short paragraph: you’re using this as a practice and reference library.
* [ ] Features (current + planned):

  * [ ] 2D geometry utilities
  * [ ] Planar 2-link forward kinematics
  * [ ] Simple serial chain model
  * [ ] Planned: Jacobian, simple IK, simple dynamics
* [ ] Installation/usage:

  * [ ] Minimal usage snippet:

    ```python
    from robotics_lib.models import planar_2link
    from robotics_lib.kinematics import fk_planar_2link
    ```
* [ ] Project structure overview:

  * [ ] 1–2 bullet explanation of each module (`geometry.py`, `kinematics.py`, etc.)
* [ ] Development section:

  * [ ] Python version requirement (e.g. “Python 3.10+”)
  * [ ] Testing command (e.g. `pytest`)

### 1.2. CONVENTIONS.md

**Goal:** Lock in terminology + style so things stay consistent and “professional”.

**Checklist:**

* [ ] **Naming & symbols**

  * [ ] `q` for joint positions (list/array of angles)
  * [ ] `qd` or `qdot` for joint velocities
  * [ ] `T` for homogeneous transforms
  * [ ] `R` for rotation matrices
  * [ ] `p`/`x` for position vectors
* [ ] **Angle conventions**

  * [ ] State clearly:

    * [ ] 2D: `theta = 0` means +x axis, positive angles are CCW
  * [ ] Units: **always radians**
* [ ] **Coordinate frames**

  * [ ] Base frame description (e.g. base at origin, x forward, y up)
  * [ ] Which frame you express FK results in (base frame).
* [ ] **Data representation**

  * [ ] Are you using `numpy` arrays for vectors/matrices? If yes:

    * [ ] All geometry functions accept/return `np.ndarray`
  * [ ] If starting pure Python, specify lists/tuples for now.
* [ ] **Type hints**

  * [ ] Commitment: “All public functions will have type hints.”
* [ ] **Docstrings**

  * [ ] Choose a style (Google/NumPy).
  * [ ] Minimal required fields for public functions: short summary, Args, Returns, Raises.
* [ ] **Error handling**

  * [ ] Guidelines: raise `ValueError` for invalid inputs (negative lengths, wrong dimensions).
  * [ ] Use `assert` only for internal invariants, not user-facing validation.

---

## 🧰 Phase 2 – Utilities & Geometry

### 2.1. `utils.py`: Helper functions

Keep this minimal to avoid “dumping ground” syndrome, but it’s useful to centralize a few cross-cutting helpers.

**Checklist:**

* [ ] Implement a small logging helper (or use `logging` directly)

  * [ ] e.g. `get_logger(name: str)` that returns a configured logger
* [ ] Implement a floating-point comparison helper (for tests & tolerances)

  * [ ] `is_close(a: float, b: float, tol: float = 1e-6) -> bool`
* [ ] (Optional) Vector type alias:

  * [ ] e.g. `Vector = Sequence[float]` or `np.ndarray` if using numpy
* [ ] (Optional) `clamp` utility:

  * [ ] `clamp(x, lo, hi)`

---

### 2.2. `geometry.py`: Vectors

If using numpy, you can lean on its ops; still good to define the API.

**Checklist:**

* [ ] Decide representation:

  * [ ] `numpy` arrays vs Python lists

* [ ] Implement vector helpers (with type hints & docstrings):

  * [ ] `add(v1, v2) -> Vec`

    * [ ] Element-wise addition
    * [ ] Check same dimension
  * [ ] `sub(v1, v2) -> Vec`

    * [ ] Element-wise subtraction
  * [ ] `dot(v1, v2) -> float`
  * [ ] `norm(v) -> float`
  * [ ] `normalize(v) -> Vec`

    * [ ] Handle zero-length vector → raise `ValueError`

* [ ] Edge case handling:

  * [ ] Wrong dimension → raise `ValueError`
  * [ ] Non-numeric → you can either trust caller or do minimal type check

* [ ] Tests for:

  * [ ] `add` basic correctness
  * [ ] `norm` and `normalize` (including zero vector error)

---

### 2.3. `geometry.py`: 2D Rotations

**Checklist:**

* [ ] Implement:

  * [ ] `rot2(theta: float) -> Mat2x2`

    * [ ] Returns 2x2 rotation matrix
  * [ ] `rotate2(v: Vec2, theta: float) -> Vec2`
* [ ] Invariants (document in docstring):

  * [ ] `rot2(0)` should be identity
  * [ ] Rotations should preserve vector length (within tolerance)
* [ ] Tests:

  * [ ] `rot2(0)` = identity
  * [ ] `rot2(pi/2)` applied to (1,0) ≈ (0,1)

---

### 2.4. `geometry.py`: 2D Homogeneous Transforms

**Checklist:**

* [ ] Implement:

  * [ ] `transform2(theta: float, tx: float, ty: float) -> Mat3x3`

    * [ ] Top-left 2x2 = `rot2(theta)`
    * [ ] Top-right 2x1 = translation
    * [ ] Last row = `[0, 0, 1]`
  * [ ] `apply_transform2(T: Mat3x3, p: Vec2) -> Vec2`

    * [ ] Convert `p` to homogeneous [x, y, 1], multiply, return xy
* [ ] Validation:

  * [ ] If `T` not 3x3 → raise `ValueError`
  * [ ] If `p` not length-2 → raise `ValueError`
* [ ] Tests:

  * [ ] Pure translation: `theta=0, tx=1, ty=2` applied to (0,0) → (1,2)
  * [ ] Pure rotation: `tx=0, ty=0` reduces to `rot2` behavior

---

## 🤖 Phase 3 – Robot Models & Kinematics (Initial)

### 3.1. `models.py`: Link dataclass

**Checklist:**

* [ ] Create:

  ```python
  @dataclass
  class Link2D:
      length: float
      # (optional now) mass: float = 0.0
      # (optional later) com: Tuple[float, float] = (0.0, 0.0)
  ```

* [ ] Validation in `__post_init__`:

  * [ ] `length >= 0` otherwise raise `ValueError`

* [ ] Tests:

  * [ ] Valid link instantiation
  * [ ] Negative length raises error

---

### 3.2. `models.py` / `kinematics.py`: SerialChain2D

You can put the class in either module; I’d lean `models.py` for data structures.

**Checklist:**

* [ ] Define:

  ```python
  @dataclass
  class SerialChain2D:
      links: Sequence[Link2D]
      # all joints assumed revolute, planar, for now
  ```
* [ ] Convenience properties:

  * [ ] `n_joints` returns `len(links)`
* [ ] Validation:

  * [ ] `len(links) >= 1`
* [ ] Tests:

  * [ ] Construct a chain with 2 links and check `n_joints == 2`

---

### 3.3. `models.py`: Helper constructor for planar 2-link

**Checklist:**

* [ ] Implement:

  ```python
  def planar_2link(l1: float, l2: float) -> SerialChain2D:
      ...
  ```
* [ ] Validate:

  * [ ] Both lengths non-negative
* [ ] Tests:

  * [ ] `planar_2link(1.0, 2.0)` yields chain with 2 links of expected lengths

---

### 3.4. `kinematics.py`: Forward Kinematics for Planar 2-Link

Start with EE position only; orientation can come later if needed.

**Checklist:**

* [ ] Implement:

  ```python
  def fk_planar_2link(l1: float, l2: float, theta1: float, theta2: float) -> Tuple[float, float]:
      """Compute (x, y) of the end-effector in base frame."""
      ...
  ```

* [ ] Document conventions:

  * [ ] Base at origin
  * [ ] First joint rotates link 1 around base
  * [ ] Second joint rotates link 2 around joint 2

* [ ] Implement a version that **doesn’t** depend on any model first (for sanity).

* [ ] Then add an overload / wrapper:

  ```python
  def fk_chain_2link(chain: SerialChain2D, q: Sequence[float]) -> Tuple[float, float]:
      ...
  ```

* [ ] Internally:

  * [ ] Use `transform2` and `apply_transform2` rather than duplicating trig.

* [ ] Tests:

  * [ ] Straight out configuration: `theta1 = 0, theta2 = 0` → (l1 + l2, 0)
  * [ ] Folded configuration: `theta1 = 0, theta2 = pi` → (l1 - l2, 0)
  * [ ] Random angles: just check that the position’s distance ≤ l1 + l2

---

### 3.5. `tests/test_kinematics.py`: FK tests with tolerances

**Checklist:**

* [ ] Use your `is_close` helper (or `pytest.approx`) to compare floats.
* [ ] Include:

  * [ ] 3–5 deterministic FK cases
  * [ ] Maybe a property-based style test:

    * [ ] For random angles, the distance to origin ≤ l1 + l2 + small epsilon

---

## 🌱 Phase 4 – (Planned for Later This Week)

You don’t have to do these today, but it’s useful to see where this is heading:

* [ ] `kinematics.py`: general N-link planar FK
* [ ] `kinematics.py`: Jacobian for planar 2-link
* [ ] `ik.py`: analytic IK for planar 2-link (elbow-up/elbow-down)
* [ ] `ik.py`: simple numeric IK (gradient or damped least squares)
* [ ] `dynamics.py`: toy 2-link dynamics (M(q), C(q, qdot), G(q))
* [ ] `tests/`: coverage for all of the above

---

If you want, next step I can:

* Turn just **Phases 0–3** into a super compact “Day 1 checklist” you can paste into your README, **or**
* Help you draft initial docstrings & function signatures (still without implementations) so you only fill in bodies.
