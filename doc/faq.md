# FAQ

## What if it doesn't run?

* Check if you have an OpenCL-enabled device on your system
  * ideally a GPU, or accelerator, since there is no attempt to optimize DeepCL for CPUs (at least, not currently, could change, feel free to submit a pull request :-) )
* Try running `gpuinfo` (from [OpenCLHelper](https://github.com/hughperkins/OpenCLHelper), but built as part of this project too, for ease of use )
  * it should output at least one OpenCL-enabled device
  * if it doesn't, then you need to make sure you have an OpenCL-enabled device, and that appropriate drivers are installed, and that the ICD is configured appropriately (registry in Windows, and `/etc/OpenCL/vendors` in linux)

## What if I need a new feature?

Please raise an issue, let me know you're interested.
* If it's on my list of things I was going to do sooner or later anyway (see below), I might do it sooner rather than later.
* If it's to do with usability, I will try to make that a priority

## What if I want to contribute myself?

- please feel free to fork this repository, tweak things, send a pull request



