import sys
import sysconfig
import site
import importlib.util
import os
import glob

print("==== PYTHON DIAGNOSTICS ====")
print("EXEC:", sys.executable)
print("VERSION:", sys.version.replace('\n', ' '))
print("PREFIX:", sys.prefix)
print("-- sys.path --")
for p in sys.path:
    print(p)

try:
    print("site.getsitepackages():", site.getsitepackages())
except Exception as e:
    print("site.getsitepackages() error:", e)

print("-- sysconfig.get_paths() --")
for k, v in sysconfig.get_paths().items():
    print(f"{k}: {v}")

spec = importlib.util.find_spec('pip')
print("importlib.util.find_spec('pip'):", spec)

print("-- try import pip --")
try:
    import pip
    print("pip module file:", getattr(pip, '__file__', None))
    print("pip.__version__:", getattr(pip, '__version__', None))
except Exception as e:
    print("import pip failed:", e)

scripts_dir = os.path.join(sys.prefix, 'Scripts')
print("Scripts dir:", scripts_dir)
if os.path.isdir(scripts_dir):
    for p in glob.glob(os.path.join(scripts_dir, 'pip*')):
        print(" -", p)
else:
    print("Scripts dir not found or not a directory")

site_packages = sysconfig.get_paths().get('purelib') or sysconfig.get_paths().get('platlib')
print("site-packages dir:", site_packages)
if site_packages and os.path.isdir(site_packages):
    for p in glob.glob(os.path.join(site_packages, 'pip*'))[:50]:
        print(" -", p)
else:
    print("site-packages dir not found or not a directory")

print("-- environment PATH (short) --")
print(os.environ.get('PATH', '')[:1000])

print("==== END DIAGNOSTICS ====")
