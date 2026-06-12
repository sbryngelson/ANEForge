#!/bin/sh
# Build the e5rt dispatch dylib that aneforge loads at runtime (one time).
# This shim wraps Apple's private e5rt framework; it links Apple frameworks, so it
# must be compiled on the target Mac. aneforge/_runtime.py loads the .dylib from
# this directory. Re-run after editing ane_e5rt_dispatch.mm.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
xcrun clang++ -O2 -Wall -Wextra -dynamiclib -fPIC -fobjc-arc \
    -framework Foundation \
    -I "$here" "$here/ane_e5rt_dispatch.mm" \
    -o "$here/libane_e5rt_dispatch.dylib"
echo "built $here/libane_e5rt_dispatch.dylib"
