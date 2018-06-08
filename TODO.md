

## Bugfixes:

- try_project_2 implementation in CUDA

## Math:

- switch from explicit euler integration to leap-frog integration


## Performance

- merge multiple calls to CUDA kernels (to avoid allocating, and copying all the time from host to kernel and vice versa)

## Other

- more README
