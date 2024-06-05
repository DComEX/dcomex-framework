<h2>Get a korali version</h1>
<pre>
git clone --quiet --single-branch git@github.com:cselab/korali.git &&
    cd korali &&
    git checkout c70d8e32258b7e2b9ed977576997dfe946816419
</pre>

<h2>Make a wheel</h2>

<pre>
sudo apt-get install -qq libeigen3-dev libgsl-dev  python3-pybind11 python3-dev ccache > /dev/null
git clone -q --depth 1 https://github.com/slitvinov/dcomex-framework.git
> dcomex-framework/korali/config.hpp
CC='ccache cc' CFLAGS='-std=c++17 -I/usr/include/eigen3' python -m pip wheel --verbose dcomex-framework/korali
</pre>
