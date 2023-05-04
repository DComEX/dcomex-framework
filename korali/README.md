# Build korali

```
meson setup build --prefix=`pwd`/tmp --buildtype=release -Donednn=false -Dmpi=true -Dopenmp=false
ninja -C build -v 2>&1 1>make.log
{ echo O = \\;  awk '/^\[[0-9]/ && sub(/^\.\./, "korali", $NF) {$0 = $NF; sub(/\.cpp$/, ".o"); sub(/\.c$/, ".o"); print $0 "\\"}' make.log | sort; } > obj.mk
```

```
git clone --depth 1 https://github.com/cselab/korali.git
make
```
