
FaceBench is built to be modular and easily extensible.

Whether you're adding a new distance metric, alignment strategy, or visualization, the process is straightforward.
This guide walks you through:

- ✅ Adding standalone functions
- 🚀 Integrating new modules into the full pipeline

---

## Add a Standalone Function

Use this when you're adding something **outside the pipeline**, like a utility, visualization, or evaluation method.

### Steps:

1. **Create the function**
   Define your logic in the appropriate submodule (`distances/`, `visualization/`, `utils/`, etc.)

2. **Expose it in `__init__.py`**
   Add your function to the root `facebench/__init__.py` so users can call it like:

   ```python
   import facebench as fb
   fb.my_custom_function(...)
   ```

3. ✅ **Done!**
   You can now use your function anywhere — with or without the pipeline.

---

## Add a New Pipeline Component

Use this if you want to create something **integrated into the full benchmarking pipeline**, like a:

- New rigid/non-rigid aligner
- New distance metric
- New mesh corrector
- New correspondence strategy
- New cropper

### Steps:

1. **Implement your function/module**
   Create it in the appropriate folder (e.g., `distances/my_method.py` or `rigid_aligners/my_icp.py`).

2. **Expose it in `__init__.py`**
   Add the function to the root `__init__.py`.

3. **Update the configuration class**
   Modify or create a new `Config` class under `config.py`
   (e.g., add a new `Enum` type like `DistanceComputerType.MY_METHOD`).

   4. **Modify the pipeline setup**
      In `pipeline.py`, locate the relevant step (e.g., "compute distance") and add your logic:

      ```python
      if dist_cfg.type == "my_method":
          errors = my_custom_distance(...)
      ```

5. ✅ **Done!**
   Your new method can now be configured and executed via `PipelineConfig`.

---

## Example

Let’s say you’re adding a new distance metric called `p2normal_distance`.

1. Add to `distances/p2normal.py`:
   ```python
   def p2normal_distance(...):
       ...
   ```

2. Expose in `__init__.py`:
   ```python
   from .distances.p2normal import p2normal_distance
   ```

3. In `config.py`, extend the enum:
   ```python
   class DistanceComputerType(str, Enum):
       ...
       P2NORMAL = "p2normal"
   ```

4. In `pipeline.py`, plug it into the dispatcher:
   ```python
   if dist_cfg.type == "p2normal"::
       errors = p2normal_distance(...)
   ```

Now you can use it in the pipeline like:

```python
fb.DistanceComputerConfig(type=fb.DistanceComputerType.P2NORMAL)
```

---

!!! note
    Every component in FaceBench follows this modular logic.
    You’re always just a few lines away from supporting your next method or experiment!

---