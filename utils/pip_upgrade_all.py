import pkg_resources
from subprocess import call

packages = [dist.project_name for dist in pkg_resources.working_set]
for package in packages:
 try:
  call("python3 -m pip install --upgrade " + package + ' --user', shell=True)
 except Exception as e:
  print(str(e))
