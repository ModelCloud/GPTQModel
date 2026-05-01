# Ascend shell startup for this 910B environment.
#
# Install this file as /etc/profile.d/ascend.sh or source it directly from a
# shell that needs CANN, OPP, torch-npu, ATB, and SiP runtime paths.

if [ -n "${CMAKE_PREFIX_PATH}" ]; then
  CMAKE_PREFIX_PATH=$(printf '%s' "${CMAKE_PREFIX_PATH}" | tr ':' '\n' | grep -v -E '^/usr/local/Ascend/cann[/_-]' | tr '\n' ':' | sed 's/:$//')
  export CMAKE_PREFIX_PATH
fi

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/cann-9.0.0-beta.2/set_env.sh ]; then
  . /usr/local/Ascend/cann-9.0.0-beta.2/set_env.sh
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  . /usr/local/Ascend/nnal/atb/set_env.sh
fi

if [ -f /usr/local/Ascend/nnal/asdsip/set_env.sh ]; then
  . /usr/local/Ascend/nnal/asdsip/set_env.sh
fi

export LD_PRELOAD=/usr/local/Ascend/cann/lib64/libjemalloc.so
